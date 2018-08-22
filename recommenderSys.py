import sys
import numpy as np


# function to read data from file.

def readData(fileName):
    results = []
    f = open(fileName, "r")
    # for each line, readFile function read the line and
    # appends the line to results list
    for line in f.readlines():
        elements = list(map(lambda x: int(x), line.split()))
        results.append(elements)
    return results


# function to form matrix using given data set.
def formMatrix(data):
    maxU = 0
    maxI = 0
    # find maximum user and item numbers to find matrix size
    for result in data:
        user = result[0]
        item = result[1]
        if (maxU < user):
            maxU = user
        if (maxI < item):
            maxI = item
    # create a numpy array(a grid of values) consisting of zeros
    matrix = np.zeros(shape=(maxU, maxI))
    # fill the numoy array
    for d in data:
        row = d[0] - 1
        col = d[1] - 1
        matrix[row, col] = d[2]
    return matrix


# function to create an output file named output.txt which is ASCII
# each line of the file will follow the format: user item rating
def outputToFile(fileName,matrix):
    f = open(fileName, "w")
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            f.write('{0} {1} {2}\n'.format(row + 1, col + 1, int(matrix[row][col])))


def stochasticGradientDescent(Q, P, userBias, itemBias, alpha, beta, trainingSet, bias):
    # SGD formula provided in README
    for row, col, rating in trainingSet:
        # predict one rating
        prediction = getSingleEntry(bias, userBias, itemBias, P, Q, row, col)
        # error between that prediction and original value
        difference = (rating - prediction)

        # biases
        # the process of factorization will converge
        # faster if biases are included in the model.
        userBias[row] += alpha * (difference - beta * userBias[row])
        itemBias[col] += alpha * (difference - beta * itemBias[col])

        # update the user and item  matrices
        P[row, :] += alpha * (difference * Q[col, :] - beta * P[row, :])
        Q[col, :] += alpha * (difference * P[row, :] - beta * Q[col, :])


# returns the approximation for all entry as matrix
# formula explained on README
def getEveryEntry(bias, userBias, itemBias, P, Q):
    matrix= bias + userBias[:, np.newaxis] + itemBias[np.newaxis:, ] + P.dot(Q.T)
    return matrix


# returns the approximation for one entry as SGD calculates items singly
# formula explained on README
def getSingleEntry(bias, userBias, itemBias, P, Q, row, col):
    prediction = bias + userBias[row] + itemBias[col] + P[row, :].dot(Q[col, :].T)
    return prediction


# function to calculate error between the estimated rating and the real rating
# I prefer to use mean squared error to manage minus errors because estimated rating can be
# either higher or lower than the real rating
def totalMeanSquareError(Q, P, matrix, bias, userBias, itemBias):
    # return the indices of elements that are nonzero
    row, col = matrix.nonzero()
    predictedMatrix = getEveryEntry(bias, userBias, itemBias, P, Q)
    error = 0
    for r, c in zip(row, col):
        # calculate error
        error += pow(matrix[r, c] - predictedMatrix[r, c], 2)
    return np.sqrt(error)


# train the algorithm to find lowest error and best bias values
def train(matrix, K):
    # First intialize the two matrices P and Q with some values,
    # calculate how different their product is to the matrix, and then try to minimize this difference iteratively.
    userNum, itemNum = matrix.shape
    P = np.random.normal(scale=1. / K, size=(userNum, K))
    Q = np.random.normal(scale=1. / K, size=(itemNum, K))
    P = np.array(P)
    Q = np.array(Q)

    # In order to better model how a rating is generated some biases may also contribute to the ratings
    # thus I added biases to my algorithm every user may have her own bias and every item may have its own bias
    itemBias = np.zeros(itemNum)
    userBias = np.zeros(userNum)
    # turn biases into numpy array
    itemBias = np.array(itemBias)
    userBias = np.array(userBias)
    # mean of all ratings
    bias = np.mean(matrix[np.where(matrix != 0)])
    bias = np.array(bias)

    # create training samples, take all the observed user-item pairs together with the ratings given
    trainingSet = [
        (row, col, matrix[row, col])
        for row in range(userNum)
        for col in range(itemNum)
        if matrix[row, col] > 0
    ]

    train = []
    # np.random.shuffle(trainingSet)
    # # calculate SGD using P and Q and get an approximation of the matrix
    # stochasticGradientDescent(Q, P, userBias, itemBias, alpha, beta, trainingSet, bias)
    # # calculate mean squared error between approximation and the real matrix
    # meanSquaredError = totalMeanSquareError(Q, P, matrix, bias, userBias, itemBias)
    i = 0
    minError = sys.maxint
    minErrorI = 0

    while i < minErrorI + 1000:
        # create a random training set
        np.random.shuffle(trainingSet)
        # calculate SGD using P and Q and get an approximate of the matrix
        stochasticGradientDescent(Q, P, userBias, itemBias, alpha, beta, trainingSet, bias)
        # calculate mean squared error between approximate and the real matrix
        meanSquaredError = totalMeanSquareError(Q, P, matrix, bias, userBias, itemBias)
        train.append((i, meanSquaredError))
        i += 1

        # Show progress
        if (i + 1) % 10 == 0:
            print("%d\t%.4f\t%.4f\t%.4f\t%d" % (
            i + 1, meanSquaredError, minError, minError - meanSquaredError, minErrorI))

        if meanSquaredError < minError:
            # Save the best result so far if there has been enough iterations since last save
            if i - minErrorI > 100:
                m = getEveryEntry(bias, userBias, itemBias, P, Q)
                print "Saving to output-%d.txt" % i
                outputToFile("output-%d.txt" % i, m)

            minError = meanSquaredError
            minErrorI = i

    # print(min(train))
    return Q, P, trainingSet, userBias, itemBias, bias


def retrieveRatings(origMatrix, approximateMatrix):
    for row in range(len(origMatrix)):
        for col in range(len(origMatrix[row])):
            if (origMatrix[row, col] > 0):
                approximateMatrix[row, col] = origMatrix[row, col]
    return approximateMatrix


def roundMatrix(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            matrix[row,col] = int(round(matrix[row,col]))
    return matrix


np.seterr(all="raise")
# create a matrix from the txt file
data = readData("train_all_txt.txt")
R = formMatrix(data)
np.array(R)


# initialize hyper parameters
K = 2  # latent features
alpha = 0.01  # rate of approaching min error-learning rate
beta = 0.01  # regularization factor


# make function calls
Q, P, trainingSet, userBias, itemBias, bias = train( R, K, )
approximateMatrix = getEveryEntry(bias, userBias, itemBias, P, Q)
retrievedMatrix = retrieveRatings(R, approximateMatrix)
roundedMatrix = roundMatrix(retrievedMatrix)

outputToFile("Aoutput.txt", approximateMatrix)
outputToFile("Routput.txt", retrievedMatrix)
outputToFile("output_.txt", roundedMatrix)

print(retrievedMatrix)
