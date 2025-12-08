import requests, pathlib

CHINOOK_DB_PATH = "dataset/sql/Chinook.db"
CHINOOK_DB_URL = f"sqlite:///{CHINOOK_DB_PATH}"

def setup_sqlite_db() :

    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    local_path = pathlib.Path(CHINOOK_DB_PATH)

    if local_path.exists():
        print(f"{local_path} already exists, skipping download.")
    else:
        response = requests.get(url)
        if response.status_code == 200:
            local_path.write_bytes(response.content)
            print(f"File downloaded and saved as {local_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")