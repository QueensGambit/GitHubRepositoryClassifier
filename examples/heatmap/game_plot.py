import matplotlib.pyplot as plt
import pandas
import numpy as np

strFile = 'C:\\Users\\MEDA-02\\Documents\\full-game'

iNRows = 45000000
id = 38

data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None, skiprows=500000 ) #,skiprows=500000

print(data)
len(data)

# only get the entries where the id is 4
dataFilt = data.loc[data[0] == id]


# .ix returns a the values from  a specific column or row


xg = data.ix[:, 2]
yg = data.ix[:, 3]
#zg = data.ix[:, 4]

xg = xg[np.logical_not(np.isnan(xg))]
yg = yg[np.logical_not(np.isnan(yg))]


x = dataFilt.ix[:, 2]
y = dataFilt.ix[:, 3]
#z = dataFilt.ix[:, 4]

#x = y[np.logical_not(np.isnan(x))]
#y = y[np.logical_not(np.isnan(y))]
print('lengthX:' + str(len(x)))

#other interesting methods:
#x.values, x.toarray, x.transpose, np.random.randn(4242)

#print('x-column only')
#print(x)


# hist2d is a non hex-represantation
#plt.hist2d(x, y, bins=50)

# add a description
fig = plt.figure(0)
fig.canvas.set_window_title('Location_Player_ID_' + str(id))
plt.hexbin(x, y, bins=None)

plt.xlim([min(xg), max(xg)])
plt.ylim([min(yg), max(yg)])

plt.title('Location of Player ID=' + str(id) + '\nwithin the first ' + str(iNRows) + ' Rows (' + str(len(x)) + 'Entries)')
plt.xlabel('X-Coordinate')
plt.ylabel('Y-Coordinate')
plt.colorbar()

#plt.clf()
plt.show()


'''
SCATTER-EXAMPLE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

plt.show()
#ax1.plot(data['x'], data['y'], color='r', label='the data')

#s

'''
