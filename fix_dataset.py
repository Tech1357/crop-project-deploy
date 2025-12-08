"""
fix_dataset.py
----------------
Purpose:
    • Read the balanced crop dataset
    • Replace soil & weather values based on scientific crop profiles
    • Assign wind speed, solar radiation, and evapotranspiration based on crop season
    • Add realistic jitter to avoid uniform synthetic data
    • Save the cleaned dataset as 'corrected_crop_dataset.csv'

Key Logic:
    1. Crop-specific profiles for:
        N, P, K, temperature, humidity, pH, rainfall
    2. Season-based ranges for:
        wind_speed_ms, solar_radiation_wm2, evapotranspiration_mm
    3. Derived features:
        organic_carbon, soil_moisture
    4. Case-insensitive matching for crop names
    5. Safe fallback ranges when crop not found

Output:
    A fully cleaned dataset ready for model training.
"""

import pandas as pd
import numpy as np
import random


def fix_dataset():

    print("Reading original dataset...")

    # Load CSV file
    try:
        df = pd.read_csv("final_balanced_crop_dataset_4600_all_districts.csv")
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    print(f"Original Data Shape: {df.shape}")
    print("Fixing soil & weather values using crop seasonal logic...")

    # -------------------------------
    # 1. Crop scientific profiles
    # -------------------------------
    crop_profiles = {
        'Rice':        [60, 90, 35, 60, 35, 45, 20, 27, 80, 89, 5.5, 7.0, 180, 300],
        'Coconut':     [20, 40, 15, 30, 25, 35, 25, 29, 90, 99, 5.0, 6.5, 130, 230],
        'Sugarcane':   [90, 120, 50, 70, 40, 60, 25, 35, 80, 90, 6.0, 7.5, 150, 220],
        'Cotton':      [100,140, 40, 60, 20, 30, 22, 29, 70, 85, 6.0, 7.5, 60, 110],
        'Maize':       [60, 90, 40, 60, 18, 25, 18, 27, 55, 70, 5.5, 7.0, 60, 100],
        'Banana':      [90,110, 70, 90, 45, 55, 25, 30, 75, 85, 5.8, 6.8, 90, 120],
        'Wheat':       [30, 50, 50, 70, 30, 45, 15, 23, 50, 70, 6.0, 7.0, 40, 80],
        'Mustard':     [20, 40, 45, 60, 20, 30, 10, 20, 30, 50, 5.5, 7.0, 30, 60],
        'Potato':      [50, 70, 40, 60, 40, 55, 12, 22, 50, 65, 5.0, 6.0, 40, 70],
        'Bengal Gram': [20, 40, 55, 75, 35, 45, 18, 25, 20, 40, 6.0, 7.5, 30, 60],
        'Toor':        [25, 45, 55, 75, 25, 35, 25, 30, 40, 60, 5.5, 7.0, 60, 90],
        'Moong':       [15, 30, 45, 65, 20, 30, 25, 32, 50, 70, 6.0, 7.2, 40, 70],
        'Urad':        [15, 30, 50, 70, 20, 30, 25, 32, 55, 75, 6.0, 7.5, 40, 75],
        'Soybean':     [30, 50, 60, 80, 35, 45, 20, 30, 40, 70, 6.0, 7.0, 50, 100],
        'Bajra':       [10, 30, 20, 40, 10, 20, 25, 35, 20, 40, 6.0, 7.5, 20, 45],
        'Sorghum':     [30, 50, 30, 50, 25, 35, 26, 34, 30, 50, 6.0, 7.0, 35, 65],
        'Ragi':        [10, 30, 20, 40, 15, 25, 26, 34, 20, 40, 5.0, 7.0, 30, 60],
        'Groundnut':   [30, 50, 40, 60, 40, 50, 24, 32, 40, 60, 5.5, 7.0, 50, 90],
        'Tobacco':     [40, 60, 30, 50, 30, 50, 22, 28, 50, 70, 5.5, 6.5, 60, 90],
        'Mirchi':      [35, 55, 50, 70, 40, 60, 20, 30, 40, 65, 5.5, 6.8, 50, 90],
        'Tomato':      [40, 60, 45, 65, 50, 70, 18, 26, 60, 80, 6.0, 7.0, 40, 90],
        'Onion':       [50, 70, 40, 60, 50, 70, 15, 25, 50, 70, 6.0, 7.0, 30, 60],
        'Sunflower':   [50, 70, 50, 70, 35, 45, 25, 30, 40, 60, 6.0, 7.5, 40, 75]
    }

    default_profile = [40, 60, 40, 60, 40, 60, 20, 30, 50, 70, 6.0, 7.0, 100, 200]

    # -------------------------------
    # 2. Crop → season mapping
    # -------------------------------
    crop_season_map = {
        'Rice':'Kharif','Cotton':'Kharif','Maize':'Kharif','Sugarcane':'Kharif',
        'Bajra':'Kharif','Soybean':'Kharif','Urad':'Kharif','Moong':'Kharif',
        'Groundnut':'Kharif','Toor':'Kharif','Ragi':'Kharif','Jute':'Kharif',
        'Wheat':'Rabi','Mustard':'Rabi','Bengal Gram':'Rabi','Potato':'Rabi',
        'Tobacco':'Rabi','Tomato':'Rabi','Onion':'Rabi',
        'Sunflower':'Zaid','Banana':'Zaid','Coconut':'Zaid','Mirchi':'Zaid'
    }

    # Season → [wind_min, wind_max, solar_min, solar_max, evap_min, evap_max]
    season_weather = {
        'Kharif':[2.5,5.5,180,220,4.0,6.0],
        'Rabi':[1.0,2.5,150,190,2.5,4.0],
        'Zaid':[3.0,6.5,230,280,6.0,8.5],
        'Default':[2.0,4.0,180,200,4.0,5.0]
    }

    # -------------------------------
    # 3. Update each row values
    # -------------------------------
    def update_row(row):
        crop = str(row['crop']).strip()

        # Get proper crop profile (case-insensitive)
        profile = crop_profiles.get(crop)
        if profile is None:
            for k in crop_profiles:
                if k.lower() == crop.lower():
                    profile = crop_profiles[k]
                    break
        if profile is None:
            profile = default_profile

        # Get season profile
        season = crop_season_map.get(crop, 'Default')
        w_profile = season_weather.get(season, season_weather['Default'])

        # Assign soil & climate values
        row['N'] = round(random.uniform(profile[0], profile[1]), 1)
        row['P'] = round(random.uniform(profile[2], profile[3]), 1)
        row['K'] = round(random.uniform(profile[4], profile[5]), 1)
        row['temperature_c'] = round(random.uniform(profile[6], profile[7]), 1)
        row['humidity_pct'] = round(random.uniform(profile[8], profile[9]), 1)
        row['pH'] = round(random.uniform(profile[10], profile[11]), 2)
        row['rainfall_mm'] = round(random.uniform(profile[12], profile[13]), 1)

        # Derived features
        row['organic_carbon'] = round(random.uniform(0.3, 0.8), 2)
        row['soil_moisture'] = round(row['humidity_pct'] * 0.45, 1)

        # Season-based features
        row['wind_speed_ms'] = round(random.uniform(w_profile[0], w_profile[1]), 2)
        row['solar_radiation_wm2'] = round(random.uniform(w_profile[2], w_profile[3]), 2)
        row['evapotranspiration_mm'] = round(random.uniform(w_profile[4], w_profile[5]), 2)

        return row


    # Apply transformations
    df_fixed = df.apply(update_row, axis=1)

    # Save output
    df_fixed.to_csv("corrected_crop_dataset.csv", index=False)
    print("\n✅ Fixed dataset saved as 'corrected_crop_dataset.csv'")
    print("✔ Soil, climatic & seasonal features successfully updated.")


if __name__ == "__main__":
    fix_dataset()
