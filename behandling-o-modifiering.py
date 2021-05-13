#Written by Nils

import csv
import pandas as pd
import numpy as np
import math
import statistics
from datetime import datetime, timedelta
import time
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Ändra working directory till rätt mapp
os.chdir('path/to/working/directory')

# Ladda in filen med svar på subjektiva frågor (som laddats ner manuellt från Google Drive)
# Den kommer vara sorterad i tidsordning när den laddas in.
df1 = pd.read_csv('frageformular.csv', sep=',', encoding= 'unicode_escape', header=1)

# Gör om datan till en matris
ans = []
for column in df1.columns: 
    li = df1[column].tolist()
    ans.append(li) 
svar = np.array(ans)

rader_svar = svar[0,:].size; kolumner_svar = svar[:,0].size

for row in range(0,rader_svar):
    for column in range(3,kolumner_svar-1):
        # Lägg in nollor istället för tomma celler.
        if svar[column,row] == 'nan':
            svar[column,row] = '0.0'    
        # Gör om svaren till att enbart innehålla siffror och inga förklaringar (bortsett från sista frågan (ja/nej))
        svar[column,row] = round(float(str(svar[column,row]).split('-')[0]))
    # Lägg in nollor istället för tomma celler.
    if (svar[kolumner_svar-1,row] == 'nan') or (svar[kolumner_svar-1,row] == 'Nej'):
        svar[kolumner_svar-1,row] = 0
    if svar[kolumner_svar-1,row] == 'Ja':
        svar[kolumner_svar-1,row] = 1


# Ladda in csv-filer med träningsdata från deltagarnas mappar i Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
fileList = drive.ListFile({'q': "sharedWithMe"}).GetList()

# Gå in i rätt mapp i Google Drive
folderID = 'yourFileIDHere' #removed actual ID before publishing
dataFolders = drive.ListFile({'q': "'%s' in parents and trashed=false" % folderID}).GetList()
samtliga = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
# Gå igenom alla deltagares mappar
for folder in dataFolders:
    singleFolder = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder['id']}).GetList()
    alldata = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
    info = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
    print(folder['title'])
    print('_______________________')
    # Gå igenom varje csv-fil i mappen
    for file in singleFolder:
        if (file['title'].endswith('.csv')) & (file['title'].startswith(folder['title'])):
            print(file['title'])
            # Ladda ner filen för att sedan kunna ladda in den i tränings-skriptet
            fileDownloaded = drive.CreateFile({'id':file['id']})
            fileDownloaded.GetContentFile('traning.csv')
            alldata = träning(alldata) #Räknar ut parametrar för träningspasset
    
    utaninfo = np.array(sorted(alldata[1:,:], key=lambda x: x[0]))
    alldata = np.vstack((info,utaninfo))
    alldata = filtrera(alldata) #Modifierar datan
    alldata = addera_pass(alldata) #Lägger ihop träningspass precis efter varandra till ett pass
    
    # Sparar ner en fil med all data för individen
    dataframe = pd.DataFrame(alldata)
    dataframe.to_csv(folder['title']+'_allt.csv', header=0)
    
    # Gör grupperingar för den individ med flest löppass + svar på subjektiva frågor
    if ('81k5' in folder['title']):
        [lopning_individ, ovrigt_individ] = separera_traning(alldata) #Dela upp löppass och icke-löppass
        dataframe = pd.DataFrame(lopning_individ)
        dataframe.to_csv('lopning_enskild.csv', header=0) #All löpning
        
        lopning_svar = hittasvar(lopning_individ)
        dataframe = pd.DataFrame(lopning_svar)
        dataframe.to_csv('lopning_svar_enskild.csv', header=0) # All löpning med subjektiva svar
    
    # Lägger in varje individs data i en större matris
    utaninfo2 = np.array(sorted(alldata[1:,:], key=lambda x: x[0]))
    samtliga = np.vstack((samtliga, utaninfo2))
   

samtliga=np.vstack((info,np.array(sorted(samtliga[1:,:], key=lambda x: x[0]))))
dataframe = pd.DataFrame(samtliga)
dataframe.to_csv('allt.csv', header=0) # All data

samtliga_svar = hittasvar(samtliga)
dataframe = pd.DataFrame(samtliga_svar)
dataframe.to_csv('samtliga_svar.csv', header=0) # All data med subjektiva svar

[lopning_alla, ovrigt_alla] = separera_traning(samtliga)
dataframe = pd.DataFrame(lopning_alla)
dataframe.to_csv('lopning_alla.csv', header=0) # All löpning

dataframe = pd.DataFrame(ovrigt_alla)
dataframe.to_csv('ovrigt_alla.csv', header=0) # Allt övrigt

lopning_svar = hittasvar(lopning_alla)
dataframe = pd.DataFrame(lopning_svar)
dataframe.to_csv('lopning_svar.csv', header=0) # All löpning med subjektiva svar

ovrigt_svar = hittasvar(ovrigt_alla)
dataframe = pd.DataFrame(ovrigt_svar)
dataframe.to_csv('ovrigt_svar.csv', header=0) # Allt övrigt med subjektiva svar


######## Funktion som ger personens maxpuls ################
def geMaxpuls():
    # Maxpuls finns i metadatan överst i varje tränings-fil
    with open('traning.csv', newline='') as f:
        metadata = list(csv.reader(f, delimiter=','))[1]
    return int(metadata[3])

######### Funktion som ger personens vikt ##################
def geVikt():
    # Vikt finns i metadatan överst i varje tränings-fil
    with open('traning.csv', newline='') as f:
        metadata = list(csv.reader(f, delimiter=','))[1]
    return float(metadata[6])

######## Funktion som räknar ut ett antal parametrar för träningspass #######
def träning(alldata):
    # Hämta den generella informationen om personen och passet
    with open('traning.csv', newline='') as f:
        metadata = list(csv.reader(f, delimiter=','))[1]
    for k in range (0,6):
        if metadata[k] == '':
            metadata[k] = 0
    
    # Namnger informationen för att lättare använda den
    kod = metadata[0]; datum = metadata[1]; träningsform = metadata[2];      
    maxpuls = int(metadata[3]); kön = metadata[4]; 
    kalorier = float(metadata[5]); vikt = float(metadata[6])
    
    # Gör träningsformen binär, 1 = löpning, 0 = övrigt
    if 'running' in träningsform.lower():
        träningsform = 1
    else:
        träningsform = 0
    
    # Gör parametern kön binär, 1 = man, 0 = kvinna
    if 'female' in kön.lower():
        kön = 0
    else:
        kön = 1
    
    # Hämta puls, distans, höjd och fart i varje tidpunkt   
    df = pd.read_csv('traning.csv', sep=',', encoding= 'unicode_escape', header=2)
    
    # Gör om träningsdatan till en matris
    v = []
    for column in df.columns: 
        li = df[column].tolist() 
        v.append(li) 
    värden=np.array(v)
    
    # Plockar ut de olika delarna av träningsdatan till olika vektorer
    pulsdata=värden[2,:]; fartdata=värden[3,:]; distansdata=värden[4,:]
    höjddata=värden[5,:]; tiddata = värden[0,:]
    
    rader = np.size(värden[0,:]) # Antal rader i matrisen
    if rader == 0:
        datum = datetime.strptime(datum.strip(), '%d/%m/%y %H:%M:%S')
        return np.vstack((alldata, np.append([datum, kod, kön, 0],np.zeros(28))))
      
    # Ta fram längden på passet i minuter (tar -1 på antal rader då det startar på sekund 0)
    tid_minuter  = (rader-1)/60
    
    # Räkna ut Medelpuls och maxpuls på passet
    trimp=0; medelpuls=0; maxpuls_pass=0; antal_fel=0
    for p in range(0,rader):
        # Om pulsdata saknas för någon tidpunkt hoppas denna över
        if pulsdata[p] == 'nan':
            pulsdata[p] = 0
            antal_fel = antal_fel + 1
        medelpuls = medelpuls + float(pulsdata[p])
        if float(pulsdata[p]) > maxpuls_pass:
            maxpuls_pass=float(pulsdata[p])
    if rader != antal_fel:
        medelpuls=medelpuls/(rader-antal_fel) # Räknar inte med de rader som saknade pulsvärde
    
    # Anpassa parametervärdena för TRIMP-uträkningen till personen beroende på kön.
    # Vilopuls från snitt bland Whoop-användare.
    # Konstant-värdet från TRIMP-formeln och norm-värdet från skalan på TRIMP-värdena för män respektive kvinnor.
    if kön == 1:
        vilopuls = 55.2;  konstant=1.92;  norm=0.436
    else:
        vilopuls = 58.8;  konstant=1.67;  norm=0.34
    
    # Räknar enbart ut trimp om det registrerats någon puls
    if medelpuls != 0:  
        for puls in pulsdata:
            if float(puls) > vilopuls: # Om pulsen är under vilopulsen (t.ex. 0) räknar vi på medelpulsen
                HR_frac = (float(puls)-vilopuls) / (maxpuls-vilopuls) # Maxpulsen är en string som behöver göras om till integer
            else:
                HR_frac = (float(medelpuls)-vilopuls) / (maxpuls-vilopuls) # Maxpulsen är en string som behöver göras om till integer
            trimp = trimp + ((1/60) * HR_frac * 0.64 * math.exp(konstant * HR_frac)/norm)

    # Räknar ut trimp/minut, för att få en skala på 0-10 vilket är mer jämförbart på vissa sätt        
    trimp_per_min = trimp/(tid_minuter)
    
    # Räkna ut VO2, energiförbrukning, distans, medeltempo och total höjdstigning.
    # Räknas endast ut om träningsformen är löpning, annars sätts värdena till 0. 
    vo2 = 0; energi = 0; stigning=0; distans = 0; medeltempo = 0
    if träningsform == 1:
        for i in range(1,rader):
            # Använder den längsta distans som registrerats under passet som passets distans
            # Är i de flesta fall det sista värdet, men i vissa fall kan värden saknas
            if (distansdata[i] != 'nan') & (float(distansdata[i]) > distans):
                distans = float(distansdata[i])
            
            # Jämför klockslagen för två på varandra följande rader
            tidskillnad = datetime.strptime(tiddata[i],"%H:%M:%S") - datetime.strptime(tiddata[i-1],"%H:%M:%S")
            if (tidskillnad.seconds == 1): #Vill inte räkna mellan två tidpunkter där klockan varit pausad (vilket motsvarar tidsskillnad > 1)
                # Räknar ut höjdstigningen under passet
                if höjddata[i] != 'nan':
                    höjdskillnad = int(höjddata[i]) - int(höjddata[i-1])
                else:
                    höjdskillnad = 0
                if (höjdskillnad > 0): #Stigningen räknas bara om höjdökningen är positiv
                    stigning = stigning + höjdskillnad
        for i in range(1,rader):
            # Vi gör enbart uträkningarna för VO2 och energiförbrukning om stigning och distans > 0
            if (i < rader-9) & (stigning > 0) & (distans > 0): 
                loopar = 0; lutning = 0; disttot = 0; hastighet = 0
                # Använder ett rullande medelvärde av 10 efterföljande värden
                for j in range (0,9):
                    # Jämför klockslagen för två på varandra följande rader
                    tiddiff = datetime.strptime(tiddata[i+j],"%H:%M:%S") - datetime.strptime(tiddata[i+j-1],"%H:%M:%S")
                    if tiddiff.seconds == 1: #Vill inte räkna mellan två tidpunkter där klockan varit pausad (vilket motsvarar tidsskillnad > 1)
                        if höjddata[i+j] != 'nan':
                            höjd = int(höjddata[i+j]) - int(höjddata[i+j-1])
                        else:
                            höjd = 0
                        if distansdata[i+j] != 'nan':
                            dist = float(distansdata[i+j]) - float(distansdata[i+j-1])
                        else:
                            dist = 0
                        disttot = dist + disttot
                        
                        if dist != 0:
                            lutning = lutning + höjd/dist # Förenklad metod för lutningen för att undvika problem man får om man försöker räkna ut den horisontella distansen (då höjd>distans, vilket är orimligt)
                        else:
                            lutning = 0
                        hastighet = hastighet + (float(fartdata[i+j]) + float(fartdata[i+j-1]))*60/2 #Hastighet i m/min
                        loopar = loopar + 1
                
                # Räkna ut medelvärden
                lutning = lutning/loopar
                hastighet = hastighet/loopar
                distsnitt = disttot/loopar
                
                ################### VO2 ##########################
                
                # Summerar VO2-värdena
                # Den vertikala komponenten av formeln räknas bara med då lutningen är positiv
                if lutning > 0:
                    vo2 = vo2 + (0.2 * hastighet + 0.9 * hastighet * lutning + 3.5)
                else:
                    vo2 = vo2 + (0.2 * hastighet + 3.5)
                
                ################# ENERGI ##########################
                
                # Energiformeln gäller bara för lutningar mellan -45% till 45%, lutningar utanför detta intervall modifieras
                if lutning > 0.45:
                    lutning = 0.45
                if lutning < -0.45:
                    lutning = -0.45
                
                # Summera alla energivärden
                energiadd = (155.4 * lutning ** 5 - 30.4 * lutning ** 4 - 43.3 * lutning ** 3 + 46.3 * lutning ** 2 + 19.5 * lutning + 3.6)
                # 3.6 är den minimala energin i varje tidpunkt (vilo-energin)
                # Multiplicera med distans i m och vikt i kg för att få energi i J
                if energiadd > 3.6: 
                    energi = energi + energiadd*distsnitt*vikt
                else:
                    energi = energi + 3.6*distsnitt*vikt
        
            
        # Räkna ut snittet av VO2 under passet
        vo2 = (vo2/(rader-10)) # Subtraktionen med 10 kommer från det rullande medelvärdet av 10 värden
                    
        # Räkna ut medeltempo i min/km       
        if (distans != 0) & (tid_minuter != 0):
            medeltempo = tid_minuter*1000/distans
    
    # Skapa ett datetime-objekt av datumet från träningspasset.
    datum_träning = datetime.strptime((datum.strip()).split(' ')[0], '%d/%m/%y')
    datum = datetime(2000+int((datum.split('/')[2]).split(' ')[0]),int(datum.split('/')[1]),int(datum.split('/')[0]), int(tiddata[rader-1].split(':')[0]), int(tiddata[rader-1].split(':')[1]), int(tiddata[rader-1].split(':')[2]))
    
    # Sammanfoga alla värden från passet till en array.
    träningsdata = np.array([datum, kod, kön, träningsform, round(tid_minuter,2), round(medelpuls,1), maxpuls_pass, round(trimp,2),round(trimp_per_min,2) , round(float(kalorier),2), round(distans,0), round(stigning,2), round(medeltempo,2), round(energi,2), round(vo2,2)])
    storlek_träningsdata = np.shape(träningsdata)[0]
    
    
    for rad in range(0,rader_svar):
        # Skapa ett datetime-objekt av datumet från svaren på de subjektiva frågorna.
        date2 = svar[0,rad].split(' ')[0]
        datum_svar = datetime(int(date2.split('-')[0]),int(date2.split('-')[1]),int(date2.split('-')[2]))
        # Om svaren kom dagen efter träningen och från personen som genomförde träningen ska dessa sammanfogas.
        if ((datum_svar-datum_träning).days == 1) & (kod == svar[1,rad]) & (np.shape(träningsdata)[0] == storlek_träningsdata):
            träningsdata = np.append(träningsdata, svar[3:,rad])
            träningsdata = np.append(träningsdata,[0,0])
        # Då matrisen med svar är sorterad efter datum kan resten av raderna hoppas över om loopen kommit så långt.
        if ((datum_svar-datum_träning).days > 1) or rad == rader_svar-1:
            # Lägga in nollor iställer för svar ifall dessa inte finns för det aktuella träningspasset.
            if np.shape(träningsdata)[0] == storlek_träningsdata:
                träningsdata = np.append(träningsdata, np.zeros(17))
            break
    
    # Lägger in passets värden till matrisen med övriga pass
    return np.vstack((alldata, träningsdata))

################ Funktion som modifierar data #########################
def filtrera(alldata):
    ### Ta fram medelvärden av olika parametrar från individens samtliga pass ###
    # Medelvärdet räknas ut exklusive nollor
    
    medelHRV = 0
    hrv = alldata[1:,26].astype('float64')
    if np.mean(hrv) > 0:
        medelHRV = np.mean(hrv[np.nonzero(hrv)])
        stdhrv = statistics.stdev(hrv[np.nonzero(hrv)])
    
    medelvilopuls = 0
    vilopuls = alldata[1:,28].astype('float64')
    if np.mean(vilopuls) > 0:
        medelvilopuls = np.mean(vilopuls[np.nonzero(vilopuls)])
        stdvilopuls = statistics.stdev(vilopuls[np.nonzero(vilopuls)])
    
    pulspass = alldata[1:,5].astype('float64')
    medelpuls = np.mean(pulspass[np.nonzero(pulspass)])
    
    maxpuls = alldata[1:,6].astype('float64')
    medel_maxpuls = np.mean(maxpuls[np.nonzero(maxpuls)])
    
    höjd = alldata[1:,11].astype('float64')
    tid = alldata[1:,4].astype('float64')
    medelhöjd = np.sum(höjd[np.nonzero(höjd)]) / np.sum(tid[np.nonzero(höjd)]) #Medelstigning i meter per min
        
    nydata = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']    
    for rad in range(1,np.size(alldata[:,0])-1):
        if (float(alldata[rad,4]) != 0): # Rader där tiden är noll tas bort
            if medelHRV != 0: # Om personen fyllt i HRV i frågeformuläret
                # Om HRV eller vilopuls saknas någon dag sätts dessa till medelvärdet
                if float(alldata[rad,26]) == 0:
                    alldata[rad,26] = round(medelHRV)
                if float(alldata[rad,28]) == 0:
                    alldata[rad,28] = round(medelvilopuls)
                # HRV-diff & vilopuls-diff = (värde-medelvärde)/standardavvikelse
                alldata[rad,30] = (float(alldata[rad,26]) - round(medelHRV))/stdhrv # HRV-diff
                alldata[rad,31] = (float(alldata[rad,28]) - round(medelvilopuls))/stdvilopuls # Vilopuls-diff
            if float(alldata[rad,5]) == 0: # Om pulsdata saknas
                alldata[rad,5] = round(medelpuls,1) # Medelpuls
                alldata[rad,6] = round(medel_maxpuls) # Maxpuls
                
                if float(alldata[rad,2]) == 1:
                    vilopuls = 55.2;  konstant=1.92;  norm=0.436
                else:
                    vilopuls = 58.8;  konstant=1.67;  norm=0.34
                
                maxpuls = geMaxpuls()
                # För uträkning av trimp sätts puls = medelpuls i formeln
                HR_frac = (float(medelpuls)-vilopuls) / (maxpuls-vilopuls) 
                # Trimp
                alldata[rad,7] = round(alldata[rad,4] * HR_frac * 0.64 * math.exp(konstant * HR_frac)/norm,2)
                # Trimp / min
                alldata[rad,8] = round(alldata[rad,7] / alldata[rad,4],2)
            
            if (float(alldata[rad,11]) == 0) & (float(alldata[rad,10]) != 0): # Om distans har registrerats men inte höjd
                alldata[rad,11] = round(medelhöjd * float(alldata[rad,4])) # Höjdstigningen räknas ut utifrån medelhöjden i m/min multiplicerat med passets längd i min 
                # Medellutning för passet, enbart sett till stigningen (de flesta pass har nettohöjdstigning 0)
                lutning = float(alldata[rad,11]) / float(alldata[rad,10]) # Förenklad metod för lutningen för att undvika problem man får om man försöker räkna ut den horisontella distansen (då höjd>distans, vilket är orimligt)
                # Energiförbrukning med medellutningen, titaldistansen och personens vikt i formeln
                alldata[rad,13] = round((155.4 * lutning ** 5 - 30.4 * lutning ** 4 - 43.3 * lutning ** 3 + 46.3 * lutning ** 2 + 19.5 * lutning + 3.6)*(float(alldata[rad,10]))*geVikt(),1)
                # VO2max uträkning med medellutning samt medelhastighet [m/min] utifrån totaldistans och totaltid
                alldata[rad,14] = round((0.2 * (float(alldata[rad,10]) / float(alldata[rad,4])) + 0.9 * (float(alldata[rad,10]) / float(alldata[rad,4])) * lutning + 3.5),2)
            # Lägger in de nya värdena tillsammans med övriga
            nydata = np.vstack((nydata, alldata[rad,:]))
    return nydata

############ Funktion som delar upp i löpträning och övrigt ###########
def separera_traning(samtliga):
    lopning = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
    ovrigt = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
    for rad in range(1,np.size(samtliga[:,0])-1):
        # Om träningsformen är löpning (1) läggs raden i löp-matrisen, annars i den andra
        if float(samtliga[rad,3]) == 1:
            lopning = np.vstack((lopning, samtliga[rad,:]))
        else:
            ovrigt = np.vstack((ovrigt, samtliga[rad,:]))
    return [lopning, ovrigt]

########## Funktion som hittar rader med formulärsvar ################
def hittasvar(data):
    nydata = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']
    if np.size(data) > np.size(nydata): # Kör loopen enbart om filen innehåller fler rader än en rad
        for rad in range(1,np.size(data[:,0])-1):
            # Om svaret på första subjektiva frågan inte är 0 så innehåller raden formulärsvar
            if float(data[rad,15]) != 0:
                nydata = np.vstack((nydata, data[rad,:]))
    return nydata

########## Funktion som lägger ihop två pass ##################
def addera_pass(data):
    nydata = ['Traning sluttid [DD/MM/YY hh:mm:ss]','Kod','Kon [1 (man)/0 (kvinna)]','Träningsform [1 (lopning)/0 (annan)]','Tid [min]','Medelpuls [bpm]','Maxpuls [bpm]','Trimp','Trimp/min [0-10]','Kalorier','Distans [m]','Hojdstigning [m]','Medeltempo [min/km]','Energi [J]', 'VO2 [mL/(kg*min)]','Traningsbelastning [1-10]','Ovrig fysisk belastning [1-10]','Muskeltrotthet [1-10]','Mental anstrangning [1-10]','Skadestatus [1-10]','Sjukdomsstatus [1-10]','Somn [1-10]','Mat- och dryck [1-10]', 'Humor [1-10]', 'Upplevd aterhamtning [1-10]', 'Motivation [1-10]', 'HRV [RMSSD ms]', 'Dagar sen mens', 'Vilopuls [bpm]','P-piller [1 (ja)/ 0 (nej)]','HRV-diff','vilopuls-diff']    
    for rad in range(2,np.size(data[:,0])):
        if (rad-1 == 1): # Lägger in den första raden av matrisen
            nydata = np.vstack((nydata, data[rad-1,:]))
        # Om passens sluttid är inom 3 timmar på samma dag läggs de samman
        if ((data[rad,0]-data[rad-1,0]).seconds < 60*60*3) & ((data[rad,0]-data[rad-1,0]).days == 0):
            if data[rad,3] == data[rad-1,3]: # Om träningsformen är densamma
                # Ny medelpuls = (medelpuls1 * tid1 + medelpuls2 * tid2) / (tid1 + tid2)
                data[rad,5] = (data[rad-1,5]*data[rad-1,4] + data[rad,5]*data[rad,4]) / (data[rad-1,4] + data[rad,4])
                # Nytt medeltempo = (medeltempo1 * tid1 + medeltempo2 * tid2) / (tid1 + tid2)
                data[rad,12] = round((data[rad-1,12]*data[rad-1,4] + data[rad,12]*data[rad,4]) / (data[rad-1,4] + data[rad,4]),2)
                # Nytt VO2 = (vo2_1 * tid1 + vo2_2 * tid2) / (tid1 + tid2)
                data[rad,14] = (data[rad-1,14]*data[rad-1,4] + data[rad,14]*data[rad,4]) / (data[rad-1,4] + data[rad,4])
                # Ny tid = tid1 + tid2
                data[rad,4] = data[rad-1,4] + data[rad,4]
                # Ny maxpuls = max(maxpuls1,maxpuls2)
                data[rad,6] = max(data[rad,6],data[rad-1,6])
                # Till uträkning av trimp
                if float(data[rad,2]) == 1:
                    vilopuls = 55.2;  konstant=1.92;  norm=0.436
                else:
                    vilopuls = 58.8;  konstant=1.67;  norm=0.34
                
                maxpuls = geMaxpuls()
                # I uträkningen av trimp används puls = medelpuls i formeln
                HR_frac = (float(data[rad,5])-vilopuls) / (maxpuls-vilopuls) 
                # Trimp
                data[rad,7] = data[rad,4] * HR_frac * 0.64 * math.exp(konstant * HR_frac)/norm
                data[rad,8] = data[rad,7] / data[rad,4] # Trimp/min
                
                # Kalori, distans, höjdstigning och energiförbrukning är bara summan av de två passens värden
                data[rad,9] = data[rad-1,9] + data[rad,9]
                data[rad,10] = data[rad-1,10] + data[rad,10]
                data[rad,11] = data[rad-1,11] + data[rad,11]
                data[rad,13] = data[rad-1,13] + data[rad,13]
            else:
                # Om ett av passen inte är löpning sätts träningsformen till övrigt
                data[rad,3] = 0
                # Ny medelpuls = (medelpuls1 * tid1 + medelpuls2 * tid2) / (tid1 + tid2)
                data[rad,5] = (data[rad-1,5]*data[rad-1,4] + data[rad,5]*data[rad,4]) / (data[rad-1,4] + data[rad,4])
                # Ny tid = tid1 + tid2
                data[rad,4] = data[rad-1,4] + data[rad,4]
                # Ny maxpuls = max(maxpuls1,maxpuls2)
                data[rad,6] = max(data[rad,6],data[rad-1,6])
                # Till trimp-uträkning
                if float(data[rad,2]) == 1:
                    vilopuls = 55.2;  konstant=1.92;  norm=0.436
                else:
                    vilopuls = 58.8;  konstant=1.67;  norm=0.34
                
                maxpuls = geMaxpuls()
                # I uträkningen av trimp används puls = medelpuls i formeln
                HR_frac = (float(data[rad,5])-vilopuls) / (maxpuls-vilopuls) 
                # Trimp
                data[rad,7] = data[rad,4] * HR_frac * 0.64 * math.exp(konstant * HR_frac)/norm
                # Trimp/min
                data[rad,8] = data[rad,7] / data[rad,4]
                # Kaloriförbrukning är summan från de två passen
                data[rad,9] = data[rad-1,9] + data[rad,9]
                # Träningsformen är övrigt så löp-parametrarna sätts till 0
                data[rad,10:15] = np.zeros(5)
            # Radera sista raden i matrisen eftersom det lagts ihop med nästa rad
            nydata = np.delete(nydata,np.size(nydata[:,0])-1,0)
        # Lägg till raden med det ihopsatta passet
        nydata = np.vstack((nydata, data[rad,:]))
    return nydata
