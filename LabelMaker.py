import os, sys, csv, math, datetime
import numpy as np
import pprint
from scipy.interpolate import interp1d

def makeLabel(data, scale, lineNr, code, labelType): #delegates to right labelMaker
    if labelType == 'SUBJECTIVE':
        return subjectiveLabel(data, scale, code, lineNr)
    elif labelType == 'HRV':
        return HRVLabel(data, scale, code, lineNr)
    elif labelType == 'TRIMP':
        return trimpLabel(data, scale, code, lineNr)
    sys.exit('Error: LABELTYPE')

def subjectiveLabel(data, scale, code, lineNr):
    for row in data[code]:
        if row[-1] == lineNr:
            m = interp1d([scale[code][1][1],scale[code][1][0],scale[code][1][2]],[3.49,1.49,0.51])
            return str(round(float(m(row[2]))))
    sys.exit('Error: subjectiveLabel')

def HRVLabel(data, scale, code, lineNr):
    for row in data[code]:
        if row[-1] == lineNr:
            m = interp1d([scale[code][2][1],scale[code][2][0],scale[code][2][2]],[3.2,1.3,0.51])
            return str(round(float(m(row[3]))))
    sys.exit('Error: HRVLabel')

def trimpLabel(data, scale, code, lineNr):
    for row in data[code]:
        if row[-1] == lineNr:
            m = interp1d([scale[code][0][1],scale[code][0][0],scale[code][0][2]],[0.51,1.3,2.8])
            return str(round(float(m(row[5]))))
    sys.exit('Error: trimpLabel')

def get_dateTime(datetime_str):
    return datetime.datetime(int(datetime_str.split(' ')[0].split('-')[0]), int(datetime_str.split(' ')[0].split('-')[1]), int(datetime_str.split(' ')[0].split('-')[2]), int(datetime_str.split(' ')[1].split(':')[0]), int(datetime_str.split(' ')[1].split(':')[1]), int(datetime_str.split(' ')[1].split(':')[2]))

def calculateBusso(data, avgTrimpDay, stabilizer):
    k1=0.031
    k2=0
    k3=0.000035
    tau1=30.8
    tau2=16.8
    tau3=2.3
                            
    sec2days = 1/(60*60*24)
    for key in data:
        firstTime = data[key][0][0]
        for a in range(stabilizer): #inserting values to stabilize k2 before the start of the real data
            fakeDate = firstTime - datetime.timedelta(days=a)
            fakeLine = [fakeDate, avgTrimpDay[key], 0, 0, 1, 0, -1]
            data[key].insert(0,fakeLine)
        for n in range(len(data[key])):
            BussoTrimp = 0
            for i in range(n):
                k2 = 0
                for j in range(i):
                    td = int((data[key][i][0]-data[key][j][0]).total_seconds())*sec2days
                    ø = (data[key][j][1] if data[key][j][4]==1 else 0.5*data[key][j][1])
                    k2 += k3*ø*math.exp(-td/tau3)

                timeDiff = int((data[key][n][0]-data[key][i][0]).total_seconds())*sec2days
                BussoTrimp += k1*data[key][i][1]*math.exp(-timeDiff/tau1)-k2*data[key][i][1]*math.exp(-timeDiff/tau2)
            data[key][n][5] = round(BussoTrimp,2)

#SUBJECTIVE, TRIMP, HRV
stabilizer = 20
LABELTYPE = 'TRIMP'

print('*********************************************************************************************************************************************')
print(LABELTYPE)
print('*********************************************************************************************************************************************')

sec2days = 1/(60*60*24)
if LABELTYPE == 'TRIMP':
    files = [f for f in os.listdir('.') if( os.path.isfile(f) and 'csv' in f)] #all csv files
else:
    files = [f for f in os.listdir('.') if( os.path.isfile(f) and 'svar' in f)] #HRV and SUBJECTIVE needs questioneer-answers to make labels

for f in files:
    with open(f, 'r') as inFile:
        print('')
        print(f)
        print('****************')

        ofile = LABELTYPE + '_Label_' + f
        reader = csv.reader(inFile)

        data = {}
        avgTrimpDay = {}
        scale = {}
        labels = {'1': 0, '2': 0, '3': 0}

        next(reader, None)
        l=1
        for line in reader: #making data-dictionary: key=code, value = [[time, trimp, subjective, hrv, bevFor, Busso, lineNr in source file], ...]
            a = [get_dateTime(line[1]), float(line[8]), float(line[25]), float(line[27]), int(line[4]), 0,l]
            try:
                data[line[2]].append(a)
            except KeyError:
                data[line[2]] = []
                data[line[2]].append(a)
            l+=1

        for key in data: #get average Trimp per day per person
            npTrimp = np.array(data[key])
            dayRange = (npTrimp[-1,0]-npTrimp[0,0]).total_seconds()*sec2days
            avgTrimpDay[key] = sum(npTrimp[:,1])/dayRange
        
        calculateBusso(data, avgTrimpDay, stabilizer) #calculates Busso writes to column 5 in data dictionary

        for key in data: #making scale-dictionary: key=code, value = [[avgBussoTrimp, minBussoTrimp, maxBussoTrimp], [avgSubjective, ...
            avgBussoTrimp = 0
            avgHrv = 0
            avgSubjective = 0

            maxSubjective = 0
            maxBussoTrimp = 0
            maxHrv = 0

            minSubjective = 1000
            minHrv = 1000
            minBussoTrimp = 10000

            for l in range(len(data[key])):
                l += stabilizer
                if l >= len(data[key]):
                    break

                avgBussoTrimp += data[key][l][5]
                avgSubjective += data[key][l][2]
                avgHrv += data[key][l][3]

                maxBussoTrimp = max(maxBussoTrimp, data[key][l][5])
                maxSubjective = max(maxSubjective, data[key][l][2])
                maxHrv = max(maxHrv, data[key][l][3])

                minBussoTrimp = min(minBussoTrimp, data[key][l][5])
                minSubjective = min(minSubjective, data[key][l][2])
                minHrv = min(minHrv, data[key][l][3])

            avgBussoTrimp = avgBussoTrimp/(l-stabilizer)
            avgSubjective = avgSubjective/(l-stabilizer)
            avgHrv = avgHrv/(l-stabilizer)

            scale[key] = [[avgBussoTrimp, minBussoTrimp, maxBussoTrimp], [avgSubjective, minSubjective, maxSubjective], [avgHrv, minHrv, maxHrv]]
        
        with open('./done/' + ofile, 'w') as outFile: #writes outfile and counts labels
            inFile.seek(0)
            writer = csv.writer(outFile, lineterminator = '\n')

            metarow = next(reader)
            metarow.append('Label [1,2,3] (högt värde == längre återhämntning behövs)')
            writer.writerow(metarow)

            rowNr = 0
            while True:
                rowNr += 1
                try:
                    line = next(reader)
                    if 'HRV' in LABELTYPE and (line[2] == '41z5' or line[2] == '57y3'):
                        continue
                    label = makeLabel(data, scale, rowNr, line[2], LABELTYPE)

                    labels[label]+=1

                    line.append(label)

                    writer.writerow(line)
                except StopIteration:
                    break
        totLabels = labels['1']+labels['2']+labels['3']
        print('1: %i = %.2f%%, 3: %i = %.2f%%, 3: %i = %.2f%%' % (labels['1'], labels['1']/totLabels*100, labels['2'], labels['2']/totLabels*100, labels['3'], labels['3']/totLabels*100))
pp = pprint.PrettyPrinter()
#pp.pprint(scale)