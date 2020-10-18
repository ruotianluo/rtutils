from __future__ import print_function

from googleapiclient import discovery
from httplib2 import Http
from oauth2client import file, client
from oauth2client.tools import argparser, run_flow
import argparse
import os

SCOPES = 'https://www.googleapis.com/auth/drive.readonly.metadata'
store = file.Storage(os.path.expanduser("~") + '/.gdrive_credential.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets(os.path.expanduser("~") + '/.rtutils_credentials.json', SCOPES)
    args = argparser.parse_args()
    args.noauth_local_webserver = True
    creds = run_flow(flow, store, args)
DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))

files = DRIVE.files().list().execute().get('files', [])
for f in files:
    print(f['name'], f['mimeType'])