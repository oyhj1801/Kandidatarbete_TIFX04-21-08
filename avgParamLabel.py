#Written by Øyvind

import csv
import numpy as np
import pprint

individual = False


inFile = './done/SUBJECTIVE_Label_samtliga_svar.csv'
with open(inFile, 'r') as f:
    #label, code, length, trimp, calories, motivaiton 
    data = [[row[-1], row[2], row[5], row[8], row[10], row[26]] for row in csv.reader(f)]
data.remove(data[0])
data = [[data[j][0], data[j][1], float(data[j][2]), float(data[j][3]), float(data[j][4]), float(data[j][5])] for j in range(len(data))]

if individual:
    avgParLabel = {'1': {}, '2': {}, '3': {}}
    dataDict = {'1': {}, '2': {}, '3': {}}


    for i in range(len(data)):
        label = data[i][0]
        code = data[i][1]
        line = [data[i][2], data[i][3], data[i][4], data[i][5]]
        try:
            dataDict[label][code].append(line)
        except KeyError:
            dataDict[label][code] = []
            dataDict[label][code].append(line)

    for key in dataDict:
        for subkey in dataDict[key]:
            npData = np.array(dataDict[key][subkey])
            #length, trimp, calories
            avgParLabel[key][subkey] = [round(sum(npData[:,0])/len(npData[:,0]),2), round(sum(npData[:,1])/len(npData[:,1]),2), round(sum(npData[:,2])/len(npData[:,2]),2), round(sum(npData[:,3])/len(npData[:,3]),2)]
    pp = pprint.PrettyPrinter()
    print('avg of: length [min], Trimp, calories, motivation per label per person in %s' %inFile)
    pp.pprint(avgParLabel)
else:
    avgParLabel = {'1': [], '2': [], '3': []}
    dataDict = {'1': [], '2': [], '3': []}

    for i in range(len(data)):
        label = data[i][0]
        line = [data[i][2], data[i][3], data[i][4], data[i][5]]
        dataDict[label].append(line)

    for key in dataDict:
        npData = np.array(dataDict[key])
        #length, trimp, calories
        avgParLabel[key] = [round(sum(npData[:,0])/len(npData[:,0]),2), round(sum(npData[:,1])/len(npData[:,1]),2), round(sum(npData[:,2])/len(npData[:,2]),2), round(sum(npData[:,3])/len(npData[:,3]),2)]
    pp = pprint.PrettyPrinter()
    print('avg of: length [min], Trimp, calories, motivation per label in %s' %inFile)
    pp.pprint(avgParLabel)