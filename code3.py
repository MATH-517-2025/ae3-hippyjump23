import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
from scipy.stats import beta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


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






parameters = np.array([[0.5, 0.5], [5,1], [1,3], [2,2], [2,5]])

n_sim = 3

n = 10000
N_ens = np.array([1,2,3,4,5,6,7,8,9,10])

h = np.zeros(10)

for i, (a,b) in enumerate(parameters):
    
    # Generate data
    X = np.random.beta(a, b, size=n)
    epsilon = np.random.normal(0, 1, size=n)

    # Compute Y
    Y = m(X) + epsilon

    # Reshape X and Y to use sklearn, they will be (1 x n) column-vectors
    X = X.reshape(-1,1) 
    Y = Y.reshape(-1,1)

    for k, N in enumerate(N_ens):
        h[k] = h_AMISE(X,Y,N)

    plt.plot(N_ens, h,label=rf"$\alpha =$ {a}, $\beta =$ {b}")

plt.legend()    
plt.show()

    