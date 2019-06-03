from oauth2client.service_account import ServiceAccountCredentials
import httplib2

# authorize with json file
def authorize(SERVICE_ACCOUNT, SERVICE_KEY):
    SCOPES = ['https://www.googleapis.com/auth/earthengine.readonly']
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_KEY,scopes=SCOPES)

    http = creds.authorize(httplib2.Http())

    return http

def authorize_using_pk12(SERVICE_ACCOUNT,SERVICE_KEY):

    return
