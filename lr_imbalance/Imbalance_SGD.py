# EECS 553 project -- SGD with logit adjusted loss

import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

def Gi(xi,yi,alpha,delta,tau,theta):
    #input: training data xi: dimension (d+1)*1, yi: 1/-1, theta: [1 w1 ... wd]
    #output: gradient for all parameters for one iteration
    f = sigmoid(theta.T@xi)
    k = delta**(-yi*tau)*np.exp(-f*yi*tau)
    gxi = alpha/(1+k)*(-yi*tau)*k*(1+np.exp(-theta.T@xi))**-2*np.exp(-theta.T@xi)*xi
    
    return gxi

def logit_lr(x_train, y_train, x_test, y_test, step_size, num_epoch,alpha,tau):
    """
    x_train - (n_train, d)
    y_train - (n_train,)
    x_test - (n_test, d)
    y_test - (n_test,)
    num_iters: how many iterations of gradient descent to perform

    """
    # initialize
    n = x_train.shape[0]
    d = x_train.shape[1]
    n_test = x_test.shape[0]

    #construct x_bar
    Xtrain = np.concatenate([np.ones([n,1]),x_train],axis=1)
    Xtest = np.concatenate([np.ones([n_test,1]),x_test],axis=1)

    #Find delta
    Phy1 = np.count_nonzero(y_train+1)
    delta = Phy1/(n-Phy1)
    
    #SGD
    theta = np.zeros([d+1,1])
    #num_epoch = 200  #number of epoches we want to run
    #step_size = 1   #step size for SGD

    for t in range (0,num_epoch):
        perm = np.random.permutation(n)
        for i in range(0,n):
            xi = Xtrain[perm[i],:].reshape(d+1,1)
            yi = y_train[perm[i]]
            gd = Gi(xi,yi,alpha,delta,tau,theta)
            theta = theta - step_size*gd
        
        #Find loss every epoch

        print(theta[0])
    
    #Test error
    acc = 0
    for i in range (0,n_test):
        f = (theta.T@Xtest[i,:].reshape(d+1,1))
        #print('f is ',f,'y is',y_test[i])
        if (f >=0 and y_test[i]>=0) or (f <0 and y_test[i]<0):
            acc +=1
        
    acc = 1-acc/n_test
    return acc

#MAIN
#Find test error
x_train = np.load("x_train.npy")    # shape (n_train, d)
x_test = np.load("x_test.npy")      # shape (n_test, d)

y_train = np.load("y_train.npy")    # shape (n_train,)
y_test = np.load("y_test.npy")        # shape (n_test,)

tau=1
alpha=1
step_size=0.2
num_epoch = 300

acc = logit_lr(x_train, y_train, x_test, y_test, step_size, num_epoch, alpha, tau)
print("test error:", acc)




