import os
from google.cloud import storage

# 🔥 ADD THIS LINE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"c:\Users\meena\Downloads\deepshield-493817-ca210739a648.json"

client = storage.Client()
bucket = client.bucket("deepfake-dataset-26")

blob = bucket.blob("deepshield_model.h5")
blob.upload_from_filename("C:/deepshield/deepshield_model.h5")

print("Old model restored!")