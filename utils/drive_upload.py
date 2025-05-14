import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def upload_to_drive(file_path):
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        file_name = os.path.basename(file_path)
        file_drive = drive.CreateFile({'title': file_name})
        file_drive.SetContentFile(file_path)
        file_drive.Upload()

        print("✅ Uploaded to Google Drive")
        return file_drive['alternateLink']
    except Exception as e:
        print(f"❌ Failed to upload to Google Drive: {e}")
        return "Upload failed"
