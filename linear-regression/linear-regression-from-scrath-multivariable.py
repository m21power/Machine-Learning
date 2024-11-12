import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
#                  Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home 
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
#| Price (1000s dollars) 
y_train = np.array([460, 232, 178])
# used to calculate the cost function of the given w and b
def cost_function(X,Y,w,b):
    m = X.shape[0] # to get the number of rows from the X-train
    cost = 0
    for i in range(m):
        f = np.dot(X[i],w) + b
        cost += (f - Y[i])**2
    cost /=(2*m)
    return cost

def compute_gradient(X, y, w, b): 
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
    return dj_db, dj_dw

def gradient_descent(w_in,b_in,X,Y,compute_grad,LearningRate):
    J_hist = []
    w = w_in[::] #avoid modifying global w within function
    b = b_in
    db,dw = compute_grad(X,Y,w,b)
    w = w - LearningRate * dw
    b = b - LearningRate * db
    return w,b

# let initialize w and b with random values
b = 785
w = np.array([ 0.4, 18, -53, -26])
epoch = 10
L = 0.01 # learning rate

for i in range(epoch):
    w,b = gradient_descent(w,b,X_train,y_train,compute_gradient,L)
def model(X,w,b):
    return np.dot(X,w) + b

