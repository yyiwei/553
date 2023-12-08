# EECS 553 project -- Read the cancer data into numpy
import numpy as np
import pandas as pd


# Function -- Prepare train and test dataset from classified raw data
#               work both for imbalanced and balanced datasets
def Prepdata(num_trainB, num_trainM, num_testB, num_testM, xdata_B, xdata_M, nB, nM, dB):
    index_1 = np.random.choice(range(nB), num_trainB+num_testB, replace=False)
    index_2 = np.random.choice(range(nM), num_trainM+num_testM, replace=False)
    # Train & Test data seperate classes
    dataB_bal = xdata_B[index_1, :]
    dataM_bal = xdata_M[index_2, :]

    index_1 = np.random.choice(
        range(num_trainB+num_testB), num_trainB, replace=False)
    index_2 = np.random.choice(
        range(num_trainM+num_testM), num_trainM, replace=False)
    # Train Test seperate
    dataB_bal_train = dataB_bal[index_1]
    dataM_bal_train = dataM_bal[index_2]
    mask1 = np.isin(np.arange(num_trainB+num_testB), index_1, invert=True)
    mask2 = np.isin(np.arange(num_trainM+num_testM), index_2, invert=True)
    dataB_bal_test = dataB_bal[mask1]
    dataM_bal_test = dataM_bal[mask2]

    # Combine classes
    train_bal = np.vstack([dataM_bal_train, dataB_bal_train])
    test_bal = np.vstack([dataM_bal_test, dataB_bal_test])

    # shuffle
    ytrain_bal = np.ones(num_trainB+num_trainM)
    ytrain_bal[np.arange(num_trainM)] = -1
    ytest_bal = np.ones(num_testB+num_testM)
    ytest_bal[np.arange(num_testM)] = -1

    index_train = np.random.shuffle(np.arange(num_trainB+num_trainM))
    xtrain = train_bal[index_train, :].reshape([num_trainB+num_trainM, dB])
    ytrain = ytrain_bal[index_train].reshape(num_trainB+num_trainM)
    index_test = np.random.shuffle(np.arange(num_testB+num_testM))
    xtest = test_bal[index_test, :].reshape([num_testB+num_testM, dB])
    ytest = ytest_bal[index_test].reshape(num_testB+num_testM)

    xtrain = np.float32(xtrain)
    xtest = np.float32(xtest)

    return xtrain, ytrain, xtest, ytest


# Read data from spreadsheet
df = pd.read_excel("data.xlsx", sheet_name="data")
data = df.to_numpy()


# Seperate into value and label
# label 'M'  -->>  -1
# label 'B'  -->>  1
xdata = data[:, 2:]
n, d = xdata.shape
print(n, d)
ydata = data[:, 1].reshape(n)
ydata[ydata == 'M'] = -1
ydata[ydata == 'B'] = 1


# Seperate into +1/-1
index_B = np.nonzero(ydata+1)
index_M = np.nonzero(ydata-1)
xdata_B = xdata[index_B, :]
xdata_B = xdata_B.reshape(xdata_B.shape[1], xdata_B.shape[2])
xdata_M = xdata[index_M, :]
xdata_M = xdata_M.reshape(xdata_M.shape[1], xdata_M.shape[2])
nB, dB = xdata_B.shape
nM, dM = xdata_M.shape
print('The number of B class is: ', nB, dB)
print('The number of M class is: ', nM, dM)


# Prepare balanced data
num_train = 150  # for each class
num_test = 50  # for each class

xtrain, ytrain, xtest, ytest = Prepdata(
    num_train, num_train, num_test, num_test, xdata_B, xdata_M, nB, nM, dB)
# save data
np.save('xtrain_bal', xtrain)
np.save('xtest_bal', xtest)
np.save('ytrain_bal', ytrain)
np.save('ytest_bal', ytest)


# Prepare imbalanced data
num_trainB = 250
num_testB = 50
num_trainM = 25
num_testM = 50

xtrain, ytrain, xtest, ytest = Prepdata(
    num_trainB, num_trainM, num_testB, num_testM, xdata_B, xdata_M, nB, nM, dB)
# save data
np.save('xtrain_im', xtrain)
np.save('xtest_im', xtest)
np.save('ytrain_im', ytrain)
np.save('ytest_im', ytest)

print(xtrain.shape)
