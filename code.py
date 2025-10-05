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
plt.xlabel("x")
plt.ylabel("Distribution")
plt.title("Plot of beta distribution VS sample data")
plt.legend()
plt.show()


# Plot the data
x_1 = np.linspace(0, 1, 100)
y_1 = m(x_1)
plt.plot(x_1,y_1,label="m(x)") 
plt.scatter(X, Y, s=10, alpha=0.6, label="Samples")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Y = m(X) with m(x) = sin((x/3+0.1)^(-1))")
plt.legend()
plt.show()

# Plot the pdf vs the data
x_2 = np.linspace(0,1,100)
y_2 = beta.pdf(x_2, a, b)
plt.plot(x_2,y_2,label="pdf")
plt.scatter(X, Y, s=10, alpha=0.6, label="Samples")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Y = m(X) with m(x) = sin((x/3+0.1)^(-1))")
plt.legend()
plt.show()

# Reshape X and Y to use sklearn, they will be (1 x n) column-vectors
X = X.reshape(-1,1) 
Y = Y.reshape(-1,1)

# Split the data into N blocks
N = 10
blocks = np.array_split(X,N)

# prepares the design matrix
poly = PolynomialFeatures(degree=4, include_bias=True) 

for k, block in enumerate(blocks):       # loop over blocks
    print(f"Block {k}:")
    
    for i, elem in enumerate(block):     # loop over elements inside a block
        print(f"  element {i} = {elem[0]}")
        
X_poly = poly.fit_transform(X) # applies the design matrix to X

# Fit linear regression
model = LinearRegression().fit(X_poly, Y)

# Get estimated coefficients
"""
pint("Estimated coefficients:", model.intercept_, model.coef_)
"""


# Plot the polynom estimator with data

x_grid = np.linspace(0, 1, 200).reshape(-1, 1)   # Beta is supported on [0,1]
x_poly = poly.transform(x_grid)
y_pred = model.predict(x_poly)

plt.scatter(X, Y, alpha=0.4, label="Data")
plt.plot(x_grid, y_pred, color="red", lw=2, label="Polynomial fit")


plt.legend()
plt.show()

