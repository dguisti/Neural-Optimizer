"""

Machine Learning Classification Problems

1.	Write a program to perform simple classification of handwritten digits (0-9) using the MNIST training dataset (see Blackboard). Print your final cost, and the F1 score.




:author - Dallin Guisti
:date - September 23, 2019
:version - Python 3.6.5

"""
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
os.system('cls')

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def main():
    LAMBDA = .001
    ALPHA = 0.00001
    BIAS = 0
    INITIAL_THETA = 0.001
    THRESHOLD = 1
    START_POINT = 0
    BATCH_SIZE = 42000
    thetas = []
    with open("MNISTtrain.csv", "r") as f:
        attributes = f.readline().split(",")[1:]
        ylist = []
        x = []
        newF = f.readlines()
        newF = newF[START_POINT:START_POINT + BATCH_SIZE]
        for lineNum, line in enumerate(newF):
            if lineNum > -1:
                splitLine = line.split(",")
                ylist.append(float(splitLine[0]))
                x.append([float(i) for i in splitLine[1:]])

        M = BATCH_SIZE
        NUMPARAMS = len(attributes)
        print(NUMPARAMS)
        #theta = np.array([BIAS] + [INITIAL_THETA for thing in range(NUMPARAMS)])
        #theta = np.reshape(theta, (NUMPARAMS + 1, 1))
        theta = np.full((1 + NUMPARAMS, 10), INITIAL_THETA)

        tempx = np.array(x)
        imgNums = 0
        for imgNum in range(imgNums):
            plt.imsave('Images/filename'+str(imgNum)+".png", np.array(tempx[imgNum + 1]).reshape(28,28), cmap=cm.gray)
            plt.imshow(np.array(tempx[imgNum + 1]).reshape(28,28))
        newarray = np.array(tempx[0]).reshape(28,28)

        ones = np.ones((M, 1))
        print(np.shape(tempx), np.shape(ones))
        data = np.hstack((ones, tempx))
        print(np.shape(data))

        """dataNew = np.transpose(data)
        for index, col in enumerate(dataNew[1:]):
            colmax = np.max(col)
            colmin = np.min(col)
            colmean = np.mean(col)
            newcol = (col - colmean)/(colmax+.000000001 - colmin)
            dataNew[index+1] = newcol
        
        data = np.transpose(dataNew)"""

    categories = range(10)
    yListRevised = np.array([1 if yvalue == categories[0] else 0 for yvalue in ylist])
    yListRevised = np.reshape(yListRevised, (M, 1))
    for iii in categories[1:]:
        yListRevised = np.hstack((yListRevised, np.reshape(np.array([1 if yvalue == iii else 0 for yvalue in ylist]), (M, 1))))

    y = yListRevised
    modelThetas = [0.1 for i in range(len(categories))]

    h = sigmoid(np.matmul(data, theta))
    J = (np.sum(-(1-y)*np.log10(1-h + 10**-10) - y*np.log10(h + 10**-10))  + (LAMBDA * np.sum(theta**2))/2)/M
    pastCost = [0.1 for _ in range(9)]
    pastCost.append(J)

    counter = 0
    while (abs(statistics.mean(pastCost) - J) >= THRESHOLD):
        counter += 1
        if counter%1000 == 0:
            print("iteration:",counter,"   Cost:",J)
        h = sigmoid(np.matmul(data, theta))
        deltaTheta = np.zeros((len(theta),1))
        regParam = np.multiply(theta,LAMBDA)
        deltaTheta = np.zeros((NUMPARAMS + 1, 10))
        for j in range(NUMPARAMS):
            theta[j] -= ALPHA/M * np.sum((h - y) * np.reshape(data[:,j],(M,1)), axis = 0) + LAMBDA/M * theta[j] # Change thetas
        J = (np.sum(-(1-y)*np.log10(1-h + 10**-10) - y*np.log10(h + 10**-10))  + (LAMBDA * np.sum(theta**2)))/M
        pastCost.append(J)
        pastCost = pastCost[1:]
    print("Counter:",counter)
    print(theta)
    h = sigmoid(np.matmul(data, theta))
    y_mean = np.sum(y)/len(y)

    print("Cost:", J)
    
    """plt.plot(y, h,"ro")
    plt.plot(y, y, "b")
    plt.show()"""
    thetas.append(theta)
    A = np.sum(h*y)
    APlusB = np.sum(h)
    APlusC = np.sum(y)
    P = A/(APlusB)
    R = A/(APlusC)
    F1 = 2 * P*R/(P + R)
    print('F1:',F1)

    print(thetas)
    #print(np.where(h[0], (np.max(h[0]))))


if __name__ == "__main__":
    main()