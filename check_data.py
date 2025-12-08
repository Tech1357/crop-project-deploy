import pandas as pd  # Import pandas for CSV handling and DataFrame operations

def check_quality():
    print("🔍 Inspecting 'corrected_crop_dataset.csv'...\n")

    # Try loading the corrected dataset
    try:
        df = pd.read_csv("corrected_crop_dataset.csv")
    except FileNotFoundError:
        # If dataset is missing, notify the user to run fix_dataset.py first
        print("❌ Error: The file 'corrected_crop_dataset.csv' does not exist yet. Run fix_dataset.py first!")
        return

    # ---------------------------------------
    # 1. Check for Empty / Missing Values
    # ---------------------------------------
    print("--- 1. Checking for Empty/Null Values ---")

    # df.isnull().sum().sum() returns total number of NaN values across the full DataFrame
    if df.isnull().sum().sum() == 0:
        print("✅ No empty cells found! Data is full.")
    else:
        print("❌ WARNING: Found empty cells!")
        print(df.isnull().sum())  # Print how many missing values per column

    # ---------------------------------------
    # 2. Crop Logic Check (Averages)
    # ---------------------------------------
    print("\n--- 2. Checking Crop Logic (Averages) ---")
    
    # Group dataset by crop and compute mean values for selected columns
    report = df.groupby('crop')[['rainfall_mm', 'N', 'wind_speed_ms', 'solar_radiation_wm2']].mean()
    
    # -------------------------
    # Check Rice logic
    # -------------------------
    # Rice → should have high rainfall (monsoon crop)
    rice_rain = report.loc['Rice', 'rainfall_mm']     # Avg rainfall for Rice
    rice_wind = report.loc['Rice', 'wind_speed_ms']   # Avg wind speed for Rice
    
    print(f"🍚 Rice Average Rain: {rice_rain:.2f} mm (Should be > 180)")
    print(f"💨 Rice Average Wind: {rice_wind:.2f} m/s (Should be ~4.0 for Monsoon)")

    # -------------------------
    # Check Cotton logic
    # -------------------------
    # Cotton → nitrogen-heavy crop
    cotton_n = report.loc['Cotton', 'N']  # Avg Nitrogen for Cotton
    print(f"🌿 Cotton Average N:  {cotton_n:.2f} (Should be > 100)")

    # -------------------------
    # Check Wheat logic
    # -------------------------
    # Wheat → winter crop → low wind conditions
    wheat_wind = report.loc['Wheat', 'wind_speed_ms']  # Avg wind for Wheat
    print(f"❄️ Wheat Average Wind: {wheat_wind:.2f} m/s (Should be < 2.5 for Winter)")

    # Final success message
    print("\n✅ If the numbers above look correct, your fix worked perfectly!")

# Script entry point
if __name__ == "__main__":
    check_quality()  # Run the quality check function
