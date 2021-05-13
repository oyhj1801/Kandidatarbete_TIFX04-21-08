#Written by Øyvind

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from zipfile import ZipFile
from os import remove
from os.path import splitext
from fit import fit2csv, csv2csv

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
folderID = 'folderIDHere' #removed actual ID before publishing

dataFolders = drive.ListFile({'q': "'%s' in parents and trashed=false" % folderID}).GetList()
for folder in dataFolders:
    print('*************************')
    print(folder['title'])
    print('-------------------------')

    singleFolder = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder['id']}).GetList()
    for file in singleFolder:
        if folder['title'] in file['title']:
            continue

        filename = file['title']
        print(filename)

        file.GetContentFile(filename)
        if 'zip' in str(splitext(filename)[1]).lower():
            with ZipFile(filename,'r') as zipObj:
                zipObj.extractall()
            remove(filename)
            filename = str(splitext(filename)[0]) + '_ACTIVITY.FIT'

        if '.csv' in str(splitext(filename)[1]).lower():
            csvfile = csv2csv(filename, folder['title'])
        elif '.fit' in str(splitext(filename)[1]).lower():
            csvfile = fit2csv(filename, folder['title'])
        else:
            remove(filename)
            continue

        try:
            uploadfile = drive.CreateFile({'parents': [{'id': '%s' % folder['id']}]})
            uploadfile.SetContentFile(csvfile)
            uploadfile.Upload()
            uploadfile.content.close()
        except:
            print('Something went wrong in Uploading of file %s in %s' %(csvfile, folder['title']))
        remove(filename)
        remove(csvfile)