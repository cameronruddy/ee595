import numpy as np

A = np.array([1, 2, 3, 4])
B = np.array([[1, 2, 3, 4]])

print("output1: {}".format(np.shape(A)))
print("output2: {}".format(np.shape(B)))
print("output3: \n", np.dot(np.reshape(A,(4,1)), B))