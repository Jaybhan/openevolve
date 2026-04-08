import os
import numpy as np
print(np.load(os.path.join(os.path.dirname(__file__), ".best_matrix.npy")))
