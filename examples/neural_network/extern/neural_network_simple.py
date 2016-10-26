# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:58:22 2016

@author: john
# more infos at
#http://stamfordresearch.com/scikit-learn-perceptron/

"""



import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import perceptron


fig, ax = plt.subplots()
ax.grid(True)

#%matplotlib inline
#get_ipython().magic('matplotlib inline')

X = [[0., 0.], [1., 1.]]
y = [[0, 1]]

#data
d = np.array([[2, 1, 2, 5, 7, 2, 3, 6, 1, 2, 5, 4, 6, 5],
              [2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]
              ])
     

#labels for each data set
t = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])

colormap = np.array(['r', 'k'])

#figure = plt.figure(figsize=(17, 9))

# plot the figure directly to the console
plt.scatter(d[0], d[1], c=colormap[t], s=40)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1,0), random_state=1)

#clf.fit(X, y)

# rotate the data
d90 = np.rot90(d)
d90 = np.rot90(d90)
d90 = np.rot90(d90)

print("d90: ")
print(d90)

# create the model
net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
net.fit(d90, t)

# print the results
print("prediction: " + str(net.predict(d90)))
print("actual: " + str(t))
print("accuracy: " + str(net.score(d90, t) * 100) + "%")

# print the function
print("the boundary is determined by: y = mx + c")
print("coeff m: " + str(net.coef_[0,0]))
print("coeff c: " + str(net.coef_[0,1]))
print(" bias: " + str(net.intercept_))

# plot the decision boundary / hyperplane
yMin, yMax = plt.ylim()
w = net.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(yMin, yMax)
yy = a * xx - (net.intercept_[0]) / w[1]

# plot the line
plt.plot(yy, xx, 'k-')

#print(X)
#print(y)

