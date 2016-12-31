import matplotlib.pyplot as plt
import pandas
import numpy as np
from scipy.stats.kde import gaussian_kde
strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'

iNRows = 3000000
id = 38

data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000 ) #,skiprows=500000

print(data)
len(data)

# only get the entries where the id is 4
dataFilt = data.loc[data[0] == id]


#whole data for min and max values
xg = data.ix[:, 2]
yg = data.ix[:, 3]
#zg = data.ix[:, 4]

#filter infinite figures otherwise exception
xg = xg[np.logical_not(np.isnan(xg))]
yg = yg[np.logical_not(np.isnan(yg))]

# xgmin = xg.min()
# xgmax = xg.max()
# ygmin = yg.min()
# ygmax = yg.max()

# .ix returns a the values from  a specific column or row

x = dataFilt.ix[:, 2]
y = dataFilt.ix[:, 3]
#z = dataFilt.ix[:, 4]

#x = y[np.logical_not(np.isnan(x))]
#y = y[np.logical_not(np.isnan(y))]
print('lengthX:' + str(len(x)))

k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[xg.min():xg.max():x.size**0.5*1j, yg.min():yg.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))


# add a description
fig = plt.figure(figsize=(9, 10))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# alpha=0.5 will make the plots semitransparent
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

ax1.set_xlim(xg.min(), xg.max())
ax1.set_ylim(yg.min(), yg.max())
ax2.set_xlim(xg.min(), xg.max())
ax2.set_ylim(yg.min(), yg.max())

# you can also overlay your soccer field
im = plt.imread('soccer2.jpg')
ax1.imshow(im, extent=[xg.min(), xg.max(), yg.min(), yg.max()], aspect='auto')
ax2.imshow(im, extent=[xg.min(), xg.max(), yg.min(), yg.max()], aspect='auto')

plt.show()