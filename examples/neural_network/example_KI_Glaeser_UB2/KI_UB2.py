# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:04:29 2016

@author: QueensGambit
"""


# same example as in the "KI-module of Prof.Dr. Glaeser"

# python beginner tut:
#    https://en.wikibooks.org/wiki/A_Beginner%27s_Python_Tutorial/Classes

# similar example at:
#http://stamfordresearch.com/scikit-learn-perceptron/

# import the perceptron-utility
from sklearn.linear_model import perceptron
# numeric-python-lib -> short np
import numpy as np
# plotting lib -> short plt
import matplotlib.pyplot as plt

# import self defined functions & classes
from myGeoMath import Straight
import perceptronAction

import random


dM = -1.0;
dC = 100;
print("~~~~~~~~~~ defining a straight ~~~~~~~~~~")
print("dM: " + str(dM))
print("dC: " + str(dC))
myStraight = Straight(-1, 100)
#myStraight = Straight(1, 0)


myStraight.printStraight()
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# set a specific seed to get always the same result
random.seed(40)


 # show a grid in the plots   
fig, ax = plt.subplots()
ax.grid(True)


# create a perceptron
net = perceptron.Perceptron(n_iter=400, eta0=1.0)
    
iN = 100
iMax = 100
print("------> trainPerceptron")
perceptronAction.trainPerceptron(net, myStraight, iN, iMax)

iNumTestPts =25
print("------> testLearningResults")
perceptronAction.testLearningResults(net, myStraight, iNumTestPts, iMax)

# add a legend with the defined labels
legend = ax.legend(loc='upper right', shadow=True)

plt.title('perceptron-example\n y = ' + str(dM)  + 'x + ' + str(dC))

plt.show()

# draw the axis-lines
#plt.axhline(0, color='gray')
#plt.axvline(0, color='gray')

# set the axis-limits
#plt.ylim(ymin=0, ymax=100)
#plt.xlim(xmin=0, xmax=100)

# export the plot as a .pdf
#plt.savefig("perceptron_data_points.pdf")
