import random
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3 / 64

# create a random variate generator using inverse-transform technique
# integrate the PD function to get the CDF: F(x) = x^4 / 256
# set F(X) = R and solve for X:  X = 4 * R**(1/4)

def F_inverse(r):
    return 4 * r**0.25


cdf_list = [F_inverse(random.random()) for _ in range(1000)]

x = np.linspace(0, 4, 1000)

plt.hist(cdf_list, density=True)
plt.plot(x, f(x))

plt.show()