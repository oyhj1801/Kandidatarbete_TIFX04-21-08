# Kandidatarbete_TIFX04-21-08
### Code written for Bachelor thesis TIFX04-21-08 at the Chalmers University of Technology spring 2021

### All the code written during the work with our Bachelor thesis.

#### Written by Øyvind:  
  ***fit.py:*** Functions used in converting and extracting and interpolating wanted data from raw watch-files into csv-files  
  ***drive.py:*** Downloading watch-files from Google Drive, calls functions from fit.py and uploads created csv-files to Google Drive  
  ***driveDel.py:*** Deletes all previously created csv-files from watch-files from Google Drive  
  ***labelMaker.py:*** Labels the datarows for ML  
  ***trainingDistribution.py:*** plots a histogram over trainingimpulse  
  ***avgParamLabel.py:*** gets the average length, calories and trimp for all sessions in each label-class both indiviually and across individuals  
  ***labelComparator.py:*** compares the labels between the 3 methods of labeling  

#### Written by Nils:
  ***behandling-o-modifiering.py:*** Calculates paramters for each session for ML for of every participants and fills in gaps in the data

#### Written by Elias:

#### Written by Maja:
 ***ML_Models.py:*** Does the ML with the labeled data and makes accuracy plots and tables
