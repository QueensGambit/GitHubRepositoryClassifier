# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:05:29 2016

@author: QueensGambit
"""

import random
# numeric-python-lib -> short np
import numpy as np
# plotting lib -> short plt
import matplotlib.pyplot as plt

def trainPerceptron(net, myStraight, iN, iMax):
    
    lstRandX = []
    lstRandY = []
    lstLabel = []

    colormap = np.array(['r', 'k'])
    
    
    # repeat 1000 time
    # from 0 to 999 (1000 is exluded)
    for i in range(0, iN):
        # randrange generates a random integer in the intervall [0, 101)
        dRandX = random.randrange(0,iMax+1)
        dRandY = random.randrange(0,iMax+1)
        
        lstRandX.append(dRandX)
        lstRandY.append(dRandY)
        
    #    print("[" + str(dRandX) + ", " + str(dRandY) + "]")
        bRet = myStraight.pointAboveOrInStraight(dRandX, dRandY)
        lstLabel.append(bRet)
        
        
    plt.scatter(lstRandX, lstRandY, c=colormap[lstLabel], s=40, label="Training-Points") #, c=colormap[t]

    
    
    # combine the to lists to one -> needed for fit()
    arrTrainPts = list(zip(lstRandY, lstRandX))
    
    #print("arrTrainPts: ")
    #print(str(arrTrainPts))
    
    print("train the perceptron with " + str(iN) + " points...")
    # fit() means learn() in the java-package
    net.fit(arrTrainPts, lstLabel)
    
    
def testLearningResults(net, myStraight, iNumTestPts, iMax):

    lstTestX = []
    lstTestY = []
    
    lstTargetRes = []
    
    for i in range(0, iNumTestPts): 
        dRandX = random.randrange(0,iMax+1)
        dRandY = random.randrange(0,iMax+1)
    
        lstTestX.append(dRandX)
        lstTestY.append(dRandY)
        
        lstTargetRes.append(myStraight.pointAboveOrInStraight(dRandX, dRandY))
     
    lstTestPts = list(zip(lstTestY, lstTestX))
    
    plt.scatter(lstTestX, lstTestY, c='blue', s=40, label="Test-Points", marker='*') #, c=colormap[t]
    
    
    # print the results
    print("prediction: " + str(net.predict(lstTestPts)))
    print("actual: " + str(lstTargetRes))
    print("accuracy: " + str(net.score(lstTestPts, lstTargetRes) * 100) + "%")
    
    # plot the decision boundary / hyperplane
    yMin, yMax = plt.ylim()
    w = net.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 100)
    yy = a * xx - (net.intercept_[0]) / w[1]
    
    # print the function
    print("the boundary is determined by: y = mx + c")
    print("coeff m: " + str(a))
    print("coeff c: " + str((net.intercept_[0]) / w[1]))
    print(" bias: " + str(net.intercept_))
    
    # plot the line
    plt.plot(yy, xx, 'k-', label='Boundary-Line')
    