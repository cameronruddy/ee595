import os,sys
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "./data2.txt"

### Import data from txt file ###
try:
    with open(FILENAME) as f:
        raw_data = [line.rstrip().split(",") for line in f]
except FileNotFoundError:
    printf("No file named {}".format(FILENAME))
    sys.exit(0)

# Enter into x and y arrays, convert from str to float #
x_data = np.array([float(data_pair[0]) for data_pair in raw_data])
y_data = np.array([float(data_pair[1]) for data_pair in raw_data])

plt.figure(0)
plt.scatter(x_data[:],
            y_data[:], 
            color="red",
            label="Raw Data")
plt.title("Raw Data from {}".format(FILENAME))
#plt.show()

### Implement Normal Equation ###
# This is taken from the book

X_b = np.c_[np.ones((len(x_data),1)), x_data]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_data)
print(theta_best)

# Add to plot
x_vec = np.linspace(min(x_data), max(x_data))
# theta_best has to be reversed, since poly1d needs
# the coefficients in reverse order for some reason
trend = np.poly1d(theta_best[::-1])

plt.plot(x_vec, 
        trend(x_vec),
        label="Linear Normal Equation Fit")
plt.legend()
#plt.show()

### Batch Gradient Descent ###

eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y_data)
    theta = theta - eta * gradients

print(theta)