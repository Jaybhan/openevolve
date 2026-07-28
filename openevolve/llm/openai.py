"""
OpenAI API interface for LLMs

This module also supports a "manual mode" (human-in-the-loop) where prompts are written
to a task queue directory and the system waits for a corresponding *.answer.json file
"""

import asyncio
import json
import logging
import os
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_display_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render messages into a single plain-text prompt for the manual UI.
    """
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        chunks.append(f"### {role}\n{content}\n")
    return "\n".join(chunks).rstrip() + "\n"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.reasoning_max_tokens = getattr(model_cfg, "reasoning_max_tokens", None)

        # Manual mode: enabled via llm.manual_mode in config.yaml
        self.manual_mode = (getattr(model_cfg, "manual_mode", False) is True)
        self.manual_queue_dir: Optional[Path] = None

        if self.manual_mode:
            qdir = getattr(model_cfg, "_manual_queue_dir", None)
            if not qdir:
                raise ValueError(
                    "Manual mode is enabled but manual_queue_dir is missing. "
                    "This should be injected by the OpenEvolve controller."
                )
            self.manual_queue_dir = Path(str(qdir)).expanduser().resolve()
            self.manual_queue_dir.mkdir(parents=True, exist_ok=True)
            self.client = None
        else:
            # Set up API client (normal mode).
            #
            # max_retries=0, NOT self.retries. generate_with_context() already
            # implements the retry policy in its own `for attempt in
            # range(retries + 1)` loop; passing the same count to the SDK
            # multiplies them, so `retries: 1` meant up to four HTTP attempts,
            # each bounded by self.timeout. With a 180s timeout that is a
            # 12-minute worst case for one iteration, and the outer
            # asyncio.wait_for cannot see or stop the inner ones — it just
            # abandons them while they keep running and billing. Retries belong
            # in exactly one place; this is not it.
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=0,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model based on model name pattern
        # This works for all endpoints (OpenAI, Azure, OptiLLM, OpenRouter, etc.)
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Reasoning token budget, if one is configured. Applies to BOTH branches:
        # the constraint it expresses — leave room for an answer — is not specific
        # to how a given provider spells its parameters.
        #
        # Sent through extra_body as OpenRouter's unified `reasoning` object,
        # because the OpenAI SDK has no first-class parameter for it.
        #
        # The budget REPLACES the effort label rather than joining it. That is
        # forced, not stylistic: sending both returns
        #   400 - Only one of "reasoning.effort" and "reasoning.max_tokens"
        #         can be specified
        # (verified against OpenRouter with claude-fable-5 and claude-opus-5).
        # An earlier version of this sent both and would have 400'd every call.
        #
        # Nothing is lost by dropping the label. effort is a coarse name for a
        # budget the provider picks for you; max_tokens states that budget
        # outright, so it expresses "think hard" at least as well and leaves no
        # ambiguity about what the ceiling actually is.
        #
        # Why a budget and not just a bigger max_tokens: on Anthropic models the
        # two share one pool, so raising max_tokens raises the thinking ceiling
        # in lockstep and the answer can still be squeezed to nothing. Only an
        # explicit reasoning cap reserves space that thinking cannot take.
        reasoning_max_tokens = kwargs.get("reasoning_max_tokens", self.reasoning_max_tokens)
        if reasoning_max_tokens is not None:
            reasoning: Dict[str, Any] = {"max_tokens": int(reasoning_max_tokens)}
            params.pop("reasoning_effort", None)
            params["extra_body"] = {**params.get("extra_body", {}), "reasoning": reasoning}

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        # Seed only makes sense for actual API calls
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None and not self.manual_mode:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        # Manual mode: no timeout unless explicitly passed by the caller
        if self.manual_mode:
            timeout = kwargs.get("timeout", None)
            return await self._manual_wait_for_answer(params, timeout=timeout)

        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        # Live visibility: prompt size is the thing that drives reasoning length,
        # and it grows as the database fills with example programs. Logged at
        # INFO (not DEBUG) so it survives worker-process log propagation.
        _msgs = params.get("messages", [])
        _pchars = sum(len(m.get("content", "") or "") for m in _msgs)
        _cap = params.get("max_tokens") or params.get("max_completion_tokens")
        # print(), not logger: this runs inside a multiprocessing worker whose
        # logging does not propagate to the parent's handlers, but whose stdout
        # does (PYTHONUNBUFFERED=1). Same channel the evaluator's output uses.
        print(
            f"-> CALL {params.get('model')} | prompt {_pchars} chars (~{_pchars//4} tok) "
            f"in {len(_msgs)} msgs | cap {_cap}",
            flush=True,
        )
        _t0 = time.time()

        # Full-prompt dump: writes exactly what the model is sent, one file per
        # call, as it happens. The database also records prompts, but only
        # flushes them at checkpoints — this is the live view. Enabled by
        # setting OPENEVOLVE_PROMPT_DUMP to a directory.
        _dump_path = None
        _dump_dir = os.environ.get("OPENEVOLVE_PROMPT_DUMP")
        if _dump_dir:
            try:
                os.makedirs(_dump_dir, exist_ok=True)
                _dump_path = os.path.join(
                    _dump_dir,
                    f"{time.strftime('%H%M%S')}_{os.getpid()}_{int(_t0*1000)%1000:03d}.txt",
                )
                with open(_dump_path, "w") as _fh:
                    _fh.write(
                        f"MODEL: {params.get('model')}\nCAP: {_cap}\n"
                        f"PROMPT: {_pchars} chars (~{_pchars//4} tok) in {len(_msgs)} msgs\n"
                    )
                    for _m in _msgs:
                        _fh.write(
                            f"\n{'='*70}\n{str(_m.get('role','?')).upper()}\n{'='*70}\n"
                            f"{_m.get('content','')}\n"
                        )
            except Exception:
                _dump_path = None

        loop = asyncio.get_event_loop()

        if _dump_path:
            # STREAMING PATH (only when dumping is on): tokens are appended to
            # the dump file as the model emits them, so the file can be tailed
            # live. Reasoning and visible content arrive on separate deltas and
            # are written to separate sections.
            def _streamed():
                _c, _r, _usage, _finish = [], [], None, None
                sp = dict(params, stream=True, stream_options={"include_usage": True})
                with open(_dump_path, "a") as fh:
                    fh.write(f"\n{'='*70}\nRESPONSE (streaming live)\n{'='*70}\n")
                    fh.flush()
                    _in_reason = False
                    for chunk in self.client.chat.completions.create(**sp):
                        if getattr(chunk, "usage", None):
                            _usage = chunk.usage
                        if not getattr(chunk, "choices", None):
                            continue
                        ch = chunk.choices[0]
                        if getattr(ch, "finish_reason", None):
                            _finish = ch.finish_reason
                        d = getattr(ch, "delta", None)
                        if d is None:
                            continue
                        rtok = getattr(d, "reasoning", None)
                        if rtok:
                            if not _in_reason:
                                fh.write("\n--- REASONING (live) ---\n"); _in_reason = True
                            _r.append(rtok); fh.write(rtok); fh.flush()
                        ctok = getattr(d, "content", None)
                        if ctok:
                            if _in_reason:
                                fh.write("\n\n--- CONTENT (live) ---\n"); _in_reason = False
                            elif not _c:
                                fh.write("\n--- CONTENT (live) ---\n")
                            _c.append(ctok); fh.write(ctok); fh.flush()
                    fh.write("\n")
                msg = types.SimpleNamespace(
                    content="".join(_c) or None, reasoning="".join(_r) or None
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg, finish_reason=_finish)],
                    usage=_usage,
                )

            response = await loop.run_in_executor(None, _streamed)
        else:
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )
        _el = time.time() - _t0
        _u = getattr(response, "usage", None)
        if False:  # streaming path already wrote the response live
            try:
                _c0 = getattr(response, "choices", None)
                _m0 = _c0[0].message if _c0 else None
                with open(_dump_path, "a") as _fh:
                    _fh.write(
                        f"\n{'='*70}\nRESPONSE  ({_el:.0f}s, "
                        f"completion_tokens={getattr(_u,'completion_tokens','?')} of {_cap}, "
                        f"finish={getattr(_c0[0],'finish_reason','?') if _c0 else '?'})\n{'='*70}\n"
                    )
                    _r = getattr(_m0, "reasoning", None) if _m0 else None
                    if _r:
                        _fh.write(f"--- REASONING ({len(_r)} chars) ---\n{_r}\n\n")
                    _fh.write(f"--- CONTENT ---\n{(_m0.content if _m0 else None)}\n")
            except Exception:
                pass

        print(
            f"<- DONE in {_el:.0f}s | completion_tokens="
            f"{getattr(_u,'completion_tokens','?')} of {_cap} | "
            f"finish={getattr(response.choices[0],'finish_reason','?') if getattr(response,'choices',None) else '?'}",
            flush=True,
        )
        # NOTE: do not rebind `logger` here — the module-level logger is used
        # earlier in this function, and a local assignment would make every
        # reference in the function local (UnboundLocalError before this line).
        logger.debug(f"API parameters: {params}")

        # An HTTP 200 carrying no usable content is a failed call, not a valid
        # result. Returning None would propagate to the iteration handler, which
        # discards the whole iteration — paid for, with no retry, because the
        # retry loop in generate_with_context() only reacts to exceptions.
        # Raising converts a silent loss into a retryable error.
        #
        # Confirmed cause: finish_reason='length'. These models emit reasoning
        # before the visible answer; when the reasoning does not converge it
        # consumes the entire token budget and the call is cut off before any
        # content is produced. So max_tokens sets the PRICE of such a failure,
        # not the quality of successes — healthy calls stop on their own well
        # under the ceiling.
        choices = getattr(response, "choices", None)
        msg = choices[0].message if choices else None
        content = msg.content if msg else None
        if content is None:
            finish = getattr(choices[0], "finish_reason", None) if choices else None
            # Reasoning lives in a separate field and is normally discarded.
            # On failure it is the only evidence of what the model was doing
            # with the budget it burned, so surface a window into it.
            reasoning = getattr(msg, "reasoning", None) if msg else None
            usage = getattr(response, "usage", None)
            rlen = len(reasoning) if reasoning else 0
            tail = repr(reasoning[-400:]) if reasoning else "<none returned>"
            raise RuntimeError(
                f"API returned no content (finish_reason={finish!r}, "
                f"reasoning_chars={rlen}, usage={usage}). "
                f"Reasoning tail: {tail}"
            )

        logger.debug(f"API response: {content}")
        return content

    async def _manual_wait_for_answer(
        self, params: Dict[str, Any], timeout: Optional[Union[int, float]]
    ) -> str:
        """
        Manual mode: write a task JSON file and poll for *.answer.json
        If timeout is provided, we respect it; otherwise we wait indefinitely
        """

        if self.manual_queue_dir is None:
            raise RuntimeError("manual_queue_dir is not initialized")

        task_id = str(uuid.uuid4())
        messages = params.get("messages", [])
        display_prompt = _build_display_prompt(messages)

        task_payload: Dict[str, Any] = {
            "id": task_id,
            "created_at": _iso_now(),
            "model": params.get("model"),
            "display_prompt": display_prompt,
            "messages": messages,
            "meta": {
                "max_tokens": params.get("max_tokens"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "reasoning_effort": params.get("reasoning_effort"),
                "verbosity": params.get("verbosity"),
            },
        }

        task_path = self.manual_queue_dir / f"{task_id}.json"
        answer_path = self.manual_queue_dir / f"{task_id}.answer.json"

        _atomic_write_json(task_path, task_payload)
        logger.info(f"[manual_mode] Task enqueued: {task_path}")

        start = time.time()
        poll_interval = 0.5

        while True:
            if answer_path.exists():
                try:
                    data = json.loads(answer_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[manual_mode] Failed to parse answer JSON for {task_id}: {e}")
                    await asyncio.sleep(poll_interval)
                    continue

                answer = str(data.get("answer") or "")
                logger.info(f"[manual_mode] Answer received for {task_id}")
                return answer

            if timeout is not None and (time.time() - start) > float(timeout):
                raise asyncio.TimeoutError(
                    f"Manual mode timed out after {timeout} seconds waiting for answer of task {task_id}"
                )

            await asyncio.sleep(poll_interval)
