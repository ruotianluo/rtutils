import argparse

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
import os


class MyDrive:
    def __init__(self):
        gauth = GoogleAuth()
        cred_file = os.path.expanduser("~") + '/.gdrive_credential.json'
        if os.path.exists(cred_file):
            gauth.LoadCredentialsFile(cred_file)
        else:
            gauth.LoadClientConfigFile(os.path.expanduser("~") + '/.rtutils_credentials.json')
            gauth.CommandLineAuth()
            # gauth.credentials = '~/.rtutils_client.json' #GoogleCredentials.get_application_default()
            gauth.SaveCredentialsFile(cred_file)

        self.drive = GoogleDrive(gauth)

    def create_folder(self, folder, remote_folder_id):
        newFolder = self.drive.CreateFile({'title': folder, "parents": [{"kind": "drive#fileLink", "id": remote_folder_id}],
                                    "mimeType": "application/vnd.google-apps.folder"})
        newFolder.Upload()
        return newFolder['id']

    def upload_folder(self, root, folder, remote_folder_id):
        # Create folder
        newFolder_id = self.create_folder(folder, remote_folder_id)
        new_root = os.path.join(root, folder)
        for item in os.listdir(new_root):
            if os.path.isfile(os.path.join(new_root, item)):
                # Upload files in this folder
                self.upload_file(new_root, item, newFolder_id)
            else:
                # Upload folder in this folder
                self.upload_folder(new_root, item, newFolder_id)
        return newFolder_id

    def upload_file(self, root, filename, remote_folder_id=None):
        if remote_folder_id == None:
            newFile = self.drive.CreateFile({"title": filename})
        else:
            newFile = self.drive.CreateFile({"title": filename,
                                             "parents": [{"kind": "drive#fileLink", "id": remote_folder_id}]})
        newFile.SetContentFile(os.path.join(root, filename))
        newFile.Upload()

    def download_file(self, download_id, download_dir):
        newFile = self.drive.CreateFile({'id': download_id})
        newFile.FetchMetadata(fetch_all=True)
        newFile.GetContentFile(os.path.join(download_dir, newFile['title']))
        return newFile

def main():
    drive = MyDrive()
    parser = argparse.ArgumentParser(description='Drive')
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_upload = subparsers.add_parser('upload', help='upload file or folder')
    parser_download = subparsers.add_parser('download', help='download file')

    parser_upload.add_argument('upload_path', type=str, help='')
    parser_upload.add_argument('remote_folder_id', type=str, nargs='?', default=None, help='')

    parser_download.add_argument('download_id', type=str, help='')
    parser_download.add_argument('download_dir', type=str, nargs='?', default='./', help='')

    args = parser.parse_args()
    # print(args)

    if args.command == 'upload':
        path = args.upload_path
        root = os.path.dirname(path)
        name = os.path.basename(path)
        if os.path.isdir(path):
            drive.upload_folder(root, name, args.remote_folder_id)
        else:
            drive.upload_file(root, name, args.remote_folder_id)
        print('Upload done')
    elif args.command == 'download':
        file = drive.download_file(args.download_id, args.download_dir)
        print('Download finished %s' %os.path.join(args.download_dir, file["title"]))

#     file_name = args.file
#     action = args.action
#     dummy = args.dummy
#     partition = args.partition
#     length = args.length
#     num_cores = args.num_cores
#     features = args.feature_constraints
#     print("Using partition {}".format(partition))
#     if dummy:
#         print("Under dummy mode")
#     if action not in allowed_actions:
#         raise ValueError(
#             "action must be one of {}, but given: {}".format(allowed_actions, action))

#     with open(file_name) as f:
#         task_dir_list = yaml.load(f)
#     for task_dir in task_dir_list:
#         if not path.isdir(task_dir):
#             raise ValueError("{} is not a valid directory".format(task_dir))
#         else:
#             task_execute(
#                 task_dir, action, length, dummy, partition, num_cores, features)

if __name__ == '__main__':
    drive = MyDrive()
    main()
    # drive.upload_folder('../content/', 'weekly_2020-01-10', '1DagXOiUK-oqBQ7lN734X1ZdxAiRnsFC1')