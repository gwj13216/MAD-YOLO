import numpy as np
print("numpy version:", np.__version__)
print("trapz available:", hasattr(np, 'trapz'))
print("trapz test:", np.trapz([1, 2, 3], [0, 1, 2]))  # 应该输出 3.0