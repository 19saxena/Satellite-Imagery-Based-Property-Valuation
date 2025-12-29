import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
retries=Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=["GET"]
)
session.mount("https://",HTTPAdapter(max_retries=retries))
MAPBOX_TOKEN="pk.eyJ1IjoicGFkZm9vdDE5IiwiYSI6ImNtamk4eXk1YzFpOTAzZHF6Z2Z3c2FzN2IifQ.e2Rt3FmF5HaDgON-lwvocw"
ZOOM=18
IMG_SIZE="256x256"
STYLE="satellite-v9"
HEADERS={
    "User-Agent":"Mozilla/5.0"
}
def fetch_images(df,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    for _,row in tqdm(df.iterrows(),total=len(df)):
        lat,lon,pid=row["lat"],row["long"],row["id"]
        url=(
            f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
            f"{lon},{lat},{ZOOM}/"
            f"{IMG_SIZE}?access_token={MAPBOX_TOKEN}"
        )
        out_path=os.path.join(out_dir, f"{pid}.png")
        if os.path.exists(out_path):
            continue
        try:
            r=session.get(
                url,
                headers=HEADERS,
                timeout=20,
                stream=True
            )
            if r.status_code==200:
                with open(out_path,"wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                print(f"Failed {pid}|Status {r.status_code}")
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {pid}: {e}")
            continue
def main():
    train=pd.read_excel("data/raw/train.xlsx")
    test=pd.read_excel("data/raw/test2.xlsx")
    fetch_images(train,"data/images/train")
    fetch_images(test,"data/images/test")
    
if __name__ == "__main__":
    main()
