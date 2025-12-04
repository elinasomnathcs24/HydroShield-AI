import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

# Step 1: Load yearly rainfall CSVs
file_list = glob.glob(r"C:\Users\Elina\Downloads\KA_IMD_rainfall_2019_2023\*.csv")  # folder with CSVs
df_list = []

for file in file_list:
    temp = pd.read_csv(file)
    if not temp.empty:
        df_list.append(temp)

if len(df_list) == 0:
    raise ValueError("No CSVs found in the folder.")

rain_df = pd.concat(df_list, ignore_index=True)

# Ensure proper datetime
rain_df['Date'] = pd.to_datetime(rain_df['Date'], errors='coerce')
rain_df = rain_df.dropna(subset=['Date'])

# Step 2: Load IBTrACS cyclone data
cyclone_df = pd.read_csv(r"C:\Users\Elina\Downloads\ibtracs.NI.list.v04r01.csv", low_memory=False)
cyclone_df.columns = cyclone_df.columns.str.strip()

# Using WMO_WIND and WMO_PRES (as present in IBTrACS)
cols_needed = ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'BASIN']
missing_cols = [c for c in cols_needed if c not in cyclone_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in IBTrACS CSV: {missing_cols}")

cyclone_df = cyclone_df[cols_needed]

# Parse dates
cyclone_df['ISO_TIME'] = pd.to_datetime(cyclone_df['ISO_TIME'], errors='coerce')
cyclone_df = cyclone_df.dropna(subset=['ISO_TIME'])

# Ensure LAT/LON/WIND/PRESSURE are numeric
cyclone_df['LAT'] = pd.to_numeric(cyclone_df['LAT'], errors='coerce')
cyclone_df['LON'] = pd.to_numeric(cyclone_df['LON'], errors='coerce')
cyclone_df['WMO_WIND'] = pd.to_numeric(cyclone_df['WMO_WIND'], errors='coerce').fillna(0)
cyclone_df['WMO_PRES'] = pd.to_numeric(cyclone_df['WMO_PRES'], errors='coerce').fillna(1013)  # default sea-level

# Filter for Karnataka region (rough lat/lon)
lat_min, lat_max = 11.5, 18.5
lon_min, lon_max = 70.5, 78.6
cyclone_df = cyclone_df[(cyclone_df['LAT'].between(lat_min, lat_max)) &
                        (cyclone_df['LON'].between(lon_min, lon_max))]

# Step 3: Merge rainfall and cyclone to create target
# Create a "cyclone_day" flag: 1 if cyclone occurred in WB, else 0
rain_df['cyclone_day'] = 0
cyclone_dates = cyclone_df['ISO_TIME'].dt.date.unique()
rain_df['cyclone_day'] = rain_df['Date'].dt.date.isin(cyclone_dates).astype(int)


# Step 4: Feature engineering
rain_df = rain_df.sort_values('Date')

# Cumulative rainfall features
rain_df['rain_3d'] = rain_df['Rainfall_mm'].rolling(3, min_periods=1).sum()
rain_df['rain_7d'] = rain_df['Rainfall_mm'].rolling(7, min_periods=1).sum()
rain_df['rain_10d'] = rain_df['Rainfall_mm'].rolling(10, min_periods=1).sum()

# Wind and pressure: merge with rainfall dataframe by date
wind_pressure = cyclone_df.groupby(cyclone_df['ISO_TIME'].dt.date)[['WMO_WIND', 'WMO_PRES']].max()
wind_pressure = wind_pressure.rename(columns={'WMO_WIND':'wind', 'WMO_PRES':'pressure'})
rain_df['wind'] = rain_df['Date'].dt.date.map(wind_pressure['wind']).fillna(0)
rain_df['pressure'] = rain_df['Date'].dt.date.map(wind_pressure['pressure']).fillna(1013)

# Step 5: Prepare features and labels
feature_cols = ['Rainfall_mm', 'rain_3d', 'rain_7d', 'rain_10d', 'wind', 'pressure']
X = rain_df[feature_cols]
y = rain_df['cyclone_day']

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Handle class imbalance with RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Step 8: Train Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train_res, y_train_res)

# Step 9: Evaluate model
y_pred = rf.predict(X_test)
if len(np.unique(y_test)) > 1:
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
else:
    y_proba = None
    auc = 'Only one class present in y_test'

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC:", auc)

import joblib
import os

downloads_folder=r"C:\Users\Elina\Downloads"
joblib.dump(rf, os.path.join(downloads_folder,"rf_cyclone_all_years.joblib"))
print("Cyclone model saved successfully!")
