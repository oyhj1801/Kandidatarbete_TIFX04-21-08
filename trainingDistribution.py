import csv
import numpy as np
import matplotlib.pyplot as plt

fileName = 'lopning_enskild.csv' #path to file

nrBins = 21
lowerFilter = 0
upperFilter = 1000

trimpIndex = 8
data = []
with open(fileName, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        data.append(round(float(line[8]),3))
data.sort()

while data[0] < lowerFilter:
    data.remove(data[0])
while data[-1] > upperFilter:
    data.remove(data[-1])

minTrimp = data[0]
maxTrimp = data[-1]
avgTrimp = sum(data)/len(data)
print(minTrimp, maxTrimp, avgTrimp)

plt.hist(data, bins = nrBins, alpha=0.5, weights=np.ones(len(data)) / len(data))
plt.show()