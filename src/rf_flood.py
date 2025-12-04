import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# STEP 1: Load all CSVs
file_list = glob.glob(r"C:\Users\Elina\Downloads\KA_IMD_rainfall_2019_2023\*.csv")  # adjust folder path
df_list = []

for f in file_list:
    tmp = pd.read_csv(f)
    # Ensure date column is datetime
    tmp['Date'] = pd.to_datetime(tmp['Date'])
    # Keep only necessary columns
    if 'Rainfall_mm' in tmp.columns:
        tmp = tmp[['Date', 'Rainfall_mm']]
    df_list.append(tmp)

# Concatenate all years
df = pd.concat(df_list, ignore_index=True)

# STEP 2: Create features
df = df.sort_values('Date').reset_index(drop=True)
df['rain'] = df['Rainfall_mm']

# Rolling cumulative rainfall
df['rain3d'] = df['rain'].rolling(3, min_periods=1).sum()
df['rain7d'] = df['rain'].rolling(7, min_periods=1).sum()
df['rain10d'] = df['rain'].rolling(10, min_periods=1).sum()

# Day of year and month (optional, helps model seasonal patterns)
df['doy'] = df['Date'].dt.dayofyear
df['month'] = df['Date'].dt.month

# STEP 3: Define flood labels
# Mark top 10% rainfall days as flood
threshold = df['rain7d'].quantile(0.90)
df['flooded'] = (df['rain7d'] >= threshold).astype(int)

print("Total flood-positive days:", df['flooded'].sum())

# STEP 4: Train Random Forest
features = ['rain', 'rain3d', 'rain7d', 'rain10d', 'doy', 'month']
X = df[features]
y = df['flooded']

# Stratified split ensures flood days exist in both train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)
rf.fit(X_train, y_train)

# STEP 5: Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("Flood Model Evaluation:\n")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_prob)
    print("AUC:", auc)
except:
    print("AUC cannot be computed (likely due to very few positive samples)")

import joblib
import os

downloads_folder=r"C:\Users\Elina\Downloads"
joblib.dump(rf, os.path.join(downloads_folder,"rf_flood_all_years.joblib"))
print("Flood model saved successfully!")

