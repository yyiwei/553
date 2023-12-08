import numpy as np


def normalize(data, mean, std):
    return (data-mean)/std


def sigmoid(a):
    # a=np.float16(a)
    return 1/(1+np.exp(-a+1e-12))


def Gi(xi, yi, alpha, delta, tau, theta):
    # input: training data xi: dimension (d+1)*1, yi: 1/-1, theta: [1 w1 ... wd]
    # output: gradient for all parameters for one iteration
    f = sigmoid(theta.T@xi)
    k = delta**(-yi*tau)*np.exp(-f*yi*tau+1e-12)
    gxi = alpha/(1+k)*(-yi*tau)*k*(1+np.exp(-theta.T@xi+1e-12))**- \
        2*np.exp(-theta.T@xi+1e-12)*xi

    return gxi


def logit_lr(x_train, y_train, x_test, y_test, step_size, num_epoch, alpha, tau):
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

    # construct x_bar
    Xtrain = np.concatenate([np.ones([n, 1]), x_train], axis=1)
    Xtest = np.concatenate([np.ones([n_test, 1]), x_test], axis=1)

    # Find delta
    Phy1 = np.count_nonzero(y_train+1)
    delta = Phy1/(n-Phy1)

    # SGD
    theta = np.zeros([d+1, 1])
    # num_epoch = 200  #number of epoches we want to run
    # step_size = 1   #step size for SGD

    for t in range(0, num_epoch):
        perm = np.random.permutation(n)
        for i in range(0, n):
            xi = Xtrain[perm[i], :].reshape(d+1, 1)
            yi = y_train[perm[i]]
            gd = Gi(xi, yi, alpha, delta, tau, theta)
            theta = theta - step_size*gd

        # Find loss every epoch

        print(theta[0])

    # Test error
    acc = 0
    for i in range(0, n_test):
        f = (theta.T@Xtest[i, :].reshape(d+1, 1))
        # print('f is ',f,'y is',y_test[i])
        if (f >= 0 and y_test[i] >= 0) or (f < 0 and y_test[i] < 0):
            acc += 1

    acc = 1-acc/n_test
    return acc


# MAIN
# Find test error
x_train = np.load("xtrain_im.npy", allow_pickle=True)    # shape (n_train, d)
x_test = np.load("xtest_im.npy", allow_pickle=True)      # shape (n_test, d)

y_train = np.load("ytrain_im.npy", allow_pickle=True)    # shape (n_train,)
y_test = np.load("ytest_im.npy", allow_pickle=True)        # shape (n_test,)

# Normalize the data
n, d = x_train.shape
mean = np.mean(x_train, axis=0).reshape([1, d])
std = np.std(x_train, axis=0).reshape([1, d])
x_train = normalize(x_train, mean, std)
x_test = normalize(x_test, mean, std)


tau = 1
alpha = 1
step_size = 0.3
num_epoch = 500

acc = logit_lr(x_train, y_train, x_test, y_test,
               step_size, num_epoch, alpha, tau)
print("test error:", acc)
