import csv

with open('./done/TRIMP_Label_samtliga_svar.csv', 'r') as t:
    with open('./done/SUBJECTIVE_Label_samtliga_svar.csv', 'r') as s:
        with open('./done/HRV_Label_samtliga_svar.csv', 'r') as h:
            trimpLabels = [[row[2], row[-1]] for row in csv.reader(t)]
            subjectiveLabels = [[row[2], row[-1]] for row in csv.reader(s)]
            hrvLabels = [[row[2], row[-1]] for row in csv.reader(h)]

trimpLabels.remove(trimpLabels[0])
subjectiveLabels.remove(subjectiveLabels[0])
hrvLabels.remove(hrvLabels[0])

agreeDict = {}
agreeDict['all'] = 0
agreeDict['only_ts'] = 0
agreeDict['only_sh'] = 0
agreeDict['only_ht'] = 0
agreeDict['none'] = 0

i=0
while i < len(trimpLabels):
    if trimpLabels[i][0] == '41z5' or trimpLabels[i][0] == '57y3': #these codes have no HRV-values
        trimpLabels.remove(trimpLabels[i])
        subjectiveLabels.remove(subjectiveLabels[i])
        i-=1
    i+=1

for i in range(len(trimpLabels)):
    if trimpLabels[i][1] == subjectiveLabels[i][1] and subjectiveLabels[i][1] == hrvLabels:
        agreeDict['all'] += 1
    elif trimpLabels[i][1] == subjectiveLabels[i][1]:
        agreeDict['only_ts'] += 1
    elif trimpLabels[i][1] == hrvLabels[i][1]:
        agreeDict['only_ht'] += 1
    elif subjectiveLabels[i][1] == hrvLabels[i][1]:
        agreeDict['only_sh'] += 1
    else:
        agreeDict['none'] += 1
print(agreeDict)