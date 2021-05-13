import sys, fitparse, math, datetime, csv
from os.path import splitext

def MMDDYY2DDMMYY(date_str):
	mm, dd, yy = map(int, date_str.split('/'))
	if dd < 10:
		dd = '0' + str(dd)
	else:
		dd = str(dd)
	if mm < 10:
		mm = '0' + str(mm)
	else:
		mm = str(mm)
	if yy < 10:
		yy = '0' + str(yy)
	else:
		yy = str(yy)
	return dd + '/' + mm + '/' + yy


def time_to_num(time_str):
	hh, mm , ss = map(int, time_str.split(':'))
	return ss + 60*(mm + 60*hh)
def num_to_time(time_int):
	hh = math.floor(time_int/3600)
	time_int = time_int - 3600*hh
	mm = math.floor(time_int/60)
	time_int = time_int - 60*mm
	ss = time_int
	if hh < 10:
		hh = '0' + str(hh)
	else:
		hh = str(hh)
	if mm < 10:
		mm = '0' + str(mm)
	else:
		mm = str(mm)
	if ss < 10:
		ss = '0' + str(ss)
	else:
		ss = str(ss)
	return hh + ':' + mm + ':' + ss


def csv2csv(fil, code):
	with open(fil, 'r') as inFile:
		csvFile = code + '_' + splitext(fil)[0] + '.csv'

		with open(csvFile, 'w') as outFile:
			reader = csv.reader(inFile)
			writer = csv.writer(outFile, lineterminator = '\n')

			next(reader, None)
			metaline = next(reader)
			next(reader, None)
			startdate = metaline[2].split('/')[0] + '/' + metaline[2].split('/')[1] + '/' + metaline[2].split('/')[2][2] + metaline[2].split('/')[2][3]

			metaheader = ['anonym kod', 'starttidspunkt DD/MM/YY hh:mm:ss', 'bevform', 'maxpuls registrert i klockan', 'kön', 'kalorier', 'vikt:']
			metadata = [code, startdate + ' ' + metaline[3], metaline[1], metaline[24], 'male', metaline[11], metaline[23]]
			dataheader = ['klockslag', 'tid [s]', 'puls [bpm]', 'pace [m/s]', 'distans [m]', 'elevation [m]:']
			writer.writerow(metaheader)
			writer.writerow(metadata)
			writer.writerow(dataheader)

			while True:
				try:
					dataline = next(reader)
				except StopIteration:
					break				
				sek = time_to_num(dataline[1])
				klockslag = num_to_time(time_to_num(metaline[3])+sek)
				try:
					pace = float(dataline[3])/3.6
				except:
					pace = 0.0
				data = [klockslag, sek, dataline[2], pace, dataline[8], dataline[6]]
				writer.writerow(data)

	return csvFile

def fit2csv(fil, code):
	ff = fitparse.FitFile(fil)
	ff.parse()

	prevTime = 0
	stopped = 0
	stopTime = 0
	maxhr = 196
	gender = 'male'
	starttime = ''
	bevform = ''
	calories = 0
	weight = 0.0
	data = {}

	for elem in ff.get_messages():
		if elem.name == 'sport':
			bevform = elem.get_values()['sport'] + '/' + elem.get_values()['sub_sport']
		elif elem.name == 'file_id':
			starttime = MMDDYY2DDMMYY(elem.get_values()['time_created'].strftime("%D")) + ' ' + elem.get_values()['time_created'].strftime("%H:%M:%S")
		elif elem.name == 'zones_target':
			if not elem.get_values()['max_heart_rate'] == None:
				maxhr = elem.get_values()['max_heart_rate']
		elif elem.name == 'user_profile':
			gender = elem.get_values()['gender']
			weight = elem.get_values()['weight']
		elif elem.name == 'session':
			if bevform == '':
				bevform = elem.get_values()['sport']
			if not elem.get_values()['total_calories'] == None:
				calories = elem.get_values()['total_calories']
		elif elem.name == 'event':
			if 'stop' in elem.get_values()['event_type']: #'stop' for suunto og 'stop_all' for garmin
				stopped = time_to_num(elem.get_values()['timestamp'].strftime("%H:%M:%S"))
			elif elem.get_values()['event_type'] == 'start':
				stopTime += time_to_num(elem.get_values()['timestamp'].strftime("%H:%M:%S")) - stopped - 1
		elif elem.name == 'record':
			time = time_to_num(elem.get_values()['timestamp'].strftime("%H:%M:%S")) - stopTime - 1

			if time-prevTime > 1:
				for i in range(1, time-prevTime):
					try:
						hr = int(data[prevTime][0] + math.floor((elem.get_values()['heart_rate'] - data[prevTime][0])*i/(time-prevTime)))
					except:
						hr = 0
					
					try:
						alt = int(data[prevTime][1] + math.floor((elem.get_values()['enhanced_altitude'] - data[prevTime][1])*i/(time-prevTime)))
					except:
						alt = 0

					try:
						speed = data[prevTime][2] + (elem.get_values()['enhanced_speed'] - data[prevTime][2])*i/(time-prevTime)
					except KeyError:
						speed = 0
					
					try:
						dist = data[prevTime][3] + (elem.get_values()['distance'] - data[prevTime][3])*i/(time-prevTime)
					except KeyError:
						dist = 0

					try:
						clock = num_to_time(time_to_num(data[prevTime][4])+i)
					except KeyError:
						clock = 'bug'

					data[prevTime+i] = [hr, alt, speed, dist, clock]
		
			try:
				hr = int(elem.get_values()['heart_rate'])
			except:
				hr = 0
			
			try:
				alt = int(elem.get_values()['enhanced_altitude'])
			except:
				alt = 0

			try:
				speed = elem.get_values()['enhanced_speed']
			except KeyError:
				speed = 0
			
			try:
				dist = elem.get_values()['distance']
			except KeyError:
				dist = 0
			
			try:
				clock = elem.get_values()['timestamp'].strftime("%H:%M:%S")
			except KeyError:
				clock = 'bug'

			data[time] = [hr, alt, speed, dist, clock]
			prevTime = time

	#print(data)
	listPuls = [item[0] for item in list(data.values())]
	listDist = [item[3] for item in list(data.values())]
	listElev = [item[1] for item in list(data.values())]
	listPace = [item[2] for item in list(data.values())]
	listTime = list(data.keys())
	listClock = [item[4] for item in list(data.values())]

	#path2csvFile = fil.replace(fil.split('/')[-1], '')
	csvFile = code + '_' + (splitext(fil.split('/')[-1])[0]) + '.csv'
	
	with open(csvFile,'w') as f:
		f.write('anonym kod, starttidspunkt DD/MM/YY hh:mm:ss, bevform, maxpuls registrert i klockan, kön, kalorier, vikt:\n')
		f.write('%s, %s, %s, %i, %s, %i, %f\n' %(code, starttime, bevform, maxhr, gender, calories, weight))
		f.write('klockslag, tid [s], puls [bpm], pace [m/s], distans [m], elevation [m]:\n')
		for index in range(len(listTime)):
			str = '%s, %i, %i, %f, %f, %i\n' %(listClock[index], listTime[index], listPuls[index], listPace[index], listDist[index], listElev[index])
			f.write(str)
	return csvFile