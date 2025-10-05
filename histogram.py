import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Seting a seed for reproducibility
np.random.seed(12345)

# Parameters for the sample/distribution
n = 1000
a, b = 2, 5   

# Define m(x)
def m(x):
    return np.sin(1.0 / (x/3.0 + 0.1))

# Generate data
X = np.random.beta(a, b, size=n)
epsilon = np.random.normal(0, 1, size=n)

# Compute Y
Y = m(X) + epsilon


#plot the pdf
x_0 = np.linspace(0, 1, 100)
y_0 = beta.pdf(x_0, a, b)

plt.hist(X, bins=50, density=True, alpha=1)
plt.plot(x_0,y_0, label="True beta distribution")
plt.xlabel("X")
plt.ylabel("Distribution")
plt.title("Plot of beta distribution VS sample data")
plt.legend()
plt.show()
