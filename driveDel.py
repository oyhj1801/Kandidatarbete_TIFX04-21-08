#Written by Øyvind

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from zipfile import ZipFile
from os import remove
from os.path import splitext

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
            print(file['title'] + ' removed')
            file.Delete()