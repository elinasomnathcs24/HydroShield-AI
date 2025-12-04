import os
import xarray as xr
import pandas as pd

# USER SETTINGS
INPUT_PATH = r"C:\Users\Elina\Downloads\RF25_ind2024_rfp25.nc"   # CHANGE THIS TO EXTRACT FOR PARTICULAR YEAR

# LATITUDES AND LONGITUDES FOR KARNATAKA
LAT_MIN, LAT_MAX = 11.5, 18.5
LON_MIN, LON_MAX = 73.0, 78.6

YEAR_START = "2024-01-01"
YEAR_END   = "2024-12-31"

OUTPUT_CSV = "KA_daily_2024.csv"

# INTERNAL FUNCTION
def load_nc_files(path):
    if os.path.isfile(path):
        if path.endswith(".nc"):
            print("✓ Single NetCDF file detected.")
            return [path]
        else:
            raise ValueError("The file is not a .nc file.")
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".nc")]
        if not files:
            raise ValueError("Folder contains no .nc files.")
        print(f"✓ Folder detected. Found {len(files)} NetCDF file(s).")
        return sorted(files)
    else:
        raise FileNotFoundError("Invalid path. File or folder does not exist.")

# MAIN PROCESS
def main():
    files = load_nc_files(INPUT_PATH)
    combined_list = []

    for file in files:
        print(f"Processing: {file}")
        ds = xr.open_dataset(file)

        # Detect coordinate names
        lat_name = "LATITUDE" if "LATITUDE" in ds else "lat"
        lon_name = "LONGITUDE" if "LONGITUDE" in ds else "lon"
        time_name = "TIME" if "TIME" in ds else "time"

        # Detect rainfall variable
        if "RAINFALL" in ds:
            rain_name = "RAINFALL"
        elif "rf" in ds:
            rain_name = "rf"
        else:
            raise KeyError("No recognized rainfall variable found (expected 'RAINFALL' or 'rf').")

        # Subset spatially to Karnataka
        ds_sub = ds.sel(
            **{lat_name: slice(LAT_MIN, LAT_MAX),
               lon_name: slice(LON_MIN, LON_MAX)}
        )

        # Filter by year
        ds_sub = ds_sub.sel(**{time_name: slice(YEAR_START, YEAR_END)})

        combined_list.append(ds_sub)

    # Combine datasets
    combined = xr.concat(combined_list, dim=time_name)

    # Convert to DataFrame (average over lat/lon)
    df = combined[rain_name].mean(dim=[lat_name, lon_name]).to_dataframe()
    df = df.reset_index()

    df.rename(columns={time_name: "Date", rain_name: "Rainfall_mm"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n")
    print("✓ Extraction Complete!")
    print(f"✓ Saved file: {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
