import matplotlib.pylab as plt
from numpy import *

data = []
for i in range(10):
	file_name = "./rmse/" + str(i) + ".txt"
	file = open(file_name,'r')
	data1 = float(file.readline())
	data2 = float(file.readline())
	data.append(data1)
	data.append(data2)

# plt.xlim(0,20)
plt.plot(data,marker= 'o')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.show()