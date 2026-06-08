import os
import kagglehub

# Set API token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_75c98d4d52f229ae00dfff89685c7608"

print("Downloading dataset using kagglehub...")
path = kagglehub.dataset_download("pankajsomkuwar/voice-dataset-catalist")
print("DOWNLOAD_PATH =", path)
