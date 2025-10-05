import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
from scipy.stats import beta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Seting a seed for reproducibility
np.random.seed(12345)

# Define m(x)
def m(x):
    return np.sin(1.0 / (x/3.0 + 0.1))


# Defining a function returning theta_22(N), sigma_hat(N) and RSS(N)
def values(X,Y,N=1):
    
    n = X.shape[0]
    
    #Split the data into N blocks
    X_blocks = np.array_split(X,N)
    Y_blocks = np.array_split(Y,N)

    # Prepares the design matrix
    poly = PolynomialFeatures(degree=4, include_bias=True) 

    # Initialize theta_22 and sigma_hat
    theta_22 = 0.0
    RSS = 0.0

    #Loop over blocks
    for k, (X_block, Y_block) in enumerate(zip(X_blocks, Y_blocks)):       
    
        # Applies the design matrix to the block in X
        X_poly = poly.fit_transform(X_block) 
    
        # Fit linear regression
        model = LinearRegression().fit(X_poly, Y_block)
    
        #Defining m_hat and its second derivative
        intercept = model.intercept_[0]
        coef = model.coef_.ravel()[1:] #ravel() makes a 1D array, [1:] ignores the first term (which is .0)

        p = Polynomial(np.concatenate(([intercept],coef)))
            
        def m_hat(x):
            return p(x)

        def m_hat_2(x):
            return p.deriv(2)(x)
    
        # Initialize the "local" theta_22 and sigma_hat
        theta_22_j = 0.0
        RSS_j = 0.0
    
        # Loop over each element in the block
        for i, (elem_X, elem_Y) in enumerate(zip(X_block,Y_block)):     
        
            # Suming over one block
            theta_22_j += m_hat_2(elem_X[0])**2
            RSS_j += (elem_Y[0]-m_hat(elem_X[0]))**2
    
        # Suming over all blocks
        theta_22 += theta_22_j
        RSS += RSS_j

    # Final answer
    theta_22 = theta_22/n
    sigma_hat = RSS/(n-5*N)

    return theta_22, sigma_hat, RSS

def supp(X): # X is always positive with beta distribution
    return np.max(X)
    
def h_AMISE(X,Y,N=1): 
    A = values(X,Y,N)
    theta, sigma, RSS = A[0], A[1], A[2] 
    
    n = X.shape[0]
    
    h = (n**(-0.2))*((35*supp(X)*sigma/theta)**(0.2))
    return h
    
def N_max(n): #Is always <= than 5
    return int(max(min(np.floor(n/20), 5), 1))
     
def N_opt(X,Y):
    n = X.shape[0]
    N_m = N_max(n)
    RSS_N_m = values(X,Y,N_m)[2] # It is RSS(N_max)
    
    
    C = np.zeros(N_m)
    for i in range(N_m): # goes from 0 to N_max-1
        C[i] = values(X,Y,i+1)[2]/(RSS_N_m/(n-5*N_m))-(n-10*(i+1))
    
    return np.argmin(C)+1

# Parameters for the sample/distribution
n = 10
   
LOL = np.zeros((16,16))
S = np.zeros((16,16))

x = np.arange(0.5,2.1,0.1)

for i, a in enumerate(x):
    for j, b in enumerate(x):
        # Generate data
        X = np.random.beta(a, b, size=n)
        epsilon = np.random.normal(0, 1, size=n)

        # Compute Y
        Y = m(X) + epsilon

        # Reshape X and Y to use sklearn, they will be (1 x n) column-vectors
        X = X.reshape(-1,1) 
        Y = Y.reshape(-1,1)
        
        N_o = N_opt(X,Y)
        
        print(f"The optimal number of blocks for alpha {round(a,2)} and beta {round(b,2)} is {N_o} and sigma hat is {values(X,Y, N_o)[1]}")
        
        LOL[i,j] = N_o
        S[i,j] = values(X,Y, N_o)[1]
        
        
print(rf"On average, our estimated $\sigma^2$ is {np.mean(S)}.")


# Making a cute image
fig, ax = plt.subplots(figsize=(10, 10))  
cax = ax.matshow(LOL, cmap='viridis')  # use a colormap

# Add colorbar
fig.colorbar(cax)

# Add title, labels, cosmetic things and a crucial comment
title = f"Heatmap of optimal blocks number"
ax.set_title(title)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")

# Set tick positions and labels
tick_positions = np.arange(len(x))  # 0, 1, ..., 15
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels([f"{v:.1f}" for v in x])
ax.set_yticklabels([f"{v:.1f}" for v in x])

plt.show()
plt.close()