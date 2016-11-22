import matplotlib.pyplot as plt
import pandas


print('Script start...')

strFile = 'G:\\Octave\\octave_sample_data\\full-game'

iNRows = 300000
data = pandas.read_csv(strFile, delimiter=',', nrows=iNRows, header=None ) #,skiprows=500000

print(data)
len(data)

# only get the entries where the id is 4
dataFilt = data.loc[data[0] == 4]


# .ix returns a the values from  a specific column or row
x = dataFilt.ix[:, 2]
y = dataFilt.ix[:, 3]
z = dataFilt.ix[:, 4]

print('lengthX:' + str(len(x)))

#other interesting methods:
#x.values, x.toarray, x.transpose, np.random.randn(4242)

#print('x-column only')
#print(x)


# hist2d is a non hex-represantation
#plt.hist2d(x, y, bins=50)

# add a description
fig = plt.figure(0)
fig.canvas.set_window_title('Location_Player_ID_4')
plt.hexbin(x, y, bins=None)
plt.title('Location of Player ID=4\nwithin the first ' + str(iNRows) + ' Rows (' + str(len(x)) + 'Entries)')
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

print('Script end.')