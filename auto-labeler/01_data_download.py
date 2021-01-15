'''
    Downloads the data from Google drive
'''
from pydrive.auth import GoogleAuth 
from pydrive.drive import GoogleDrive 
from oauth2client.client import GoogleCredentials 

url = "https://drive.google.com/file/d/1tKxbJeMlJU_Dh62Xqgqdi7ua-AQRFEM9/view"
    
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
# gauth.LoadClientConfigFile(r'C:\Users\Paht\OneDrive\Documents\google_drive_client_secret.json')
drive = GoogleDrive(gauth)

id = url.split("/")[-2] 
 
downloaded = drive.CreateFile({'id':id})  
downloaded.GetContentFile('data/labeled_abstracts_for_ML2.csv') 

