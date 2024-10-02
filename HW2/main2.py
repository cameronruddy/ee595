import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100,1)
y = 4 + 3 *np.random.rand(100,1)
X_b = np.c_[np.ones((100, 1)),X]

print("{}\n\n{}\n\n{}\n\n".format(X,y,X_b))