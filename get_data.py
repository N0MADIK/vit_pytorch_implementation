import os 
import requests
import zipfile
from pathlib import Path

#setup path to dat folder 
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#if image folder doesn't exist download it and prepare it 
if image_path.is_dir():
    print(f"{image_path} directory exists")
else:
    print(f"did not find {image_path} directory, creating now...")
    image_path.mkdir(parents=True, exist_ok=True)

#download pizza steak sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza steak sushi dataset")
    f.write(request.content)

#unzip data file 
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping data zipfile")
    zip_ref.extractall(image_path)

#Remove zipfile
os.remove(data_path / "pizza_steak_sushi.zip")