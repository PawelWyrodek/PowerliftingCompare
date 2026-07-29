import pandas as pd
import requests
import zipfile
import io

URL = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"

print("Pobieranie archiwum OpenPowerlifting...")
response = requests.get(URL)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # Pobieramy nazwę pliku CSV z wnętrza zipa
    csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
    print(f"Rozpakowywanie {csv_filename}...")
    
    # Odczytujemy CSV z optymalizacją typów danych (saves RAM)
    df = pd.read_csv(z.open(csv_filename), low_memory=False)

print("Konwersja do formatu Parquet...")
# Zapisujemy do mocno skompresowanego formatu parquet
df.to_parquet("openpowerlifting.parquet", engine="pyarrow", compression="snappy")
print("Gotowe!")