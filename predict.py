import joblib
import numpy as np
import pandas as pd

def predict_crop():
    print("⏳ Loading Model and Encoder...")
    
    try:
        model = joblib.load('crop_model_final.pkl')
        encoder = joblib.load('label_encoder.pkl')
        print("✅ System Ready!")
    except FileNotFoundError:
        print("❌ Error: Files not found. Run main.ipynb first.")
        return

    feature_names = [
        'N', 'P', 'K', 'temperature_c', 'humidity_pct', 'pH', 'rainfall_mm',
        'organic_carbon', 'soil_moisture', 
        'wind_speed_ms', 'solar_radiation_wm2', 'evapotranspiration_mm'
    ]

    print("\n🌾 --- INTELLIGENT CROP PREDICTOR (Full Input Mode) --- 🌾")
    
    try:
        # --- PRIMARY INPUTS ---
        print("--- Soil Nutrients ---")
        n = float(input("1. Nitrogen (N) [10-140]: "))
        p = float(input("2. Phosphorus (P) [10-100]: "))
        k = float(input("3. Potassium (K) [10-100]: "))
        
        print("\n--- Basic Weather & Soil ---")
        temp = float(input("4. Temperature (°C) [10-40]: "))
        humid = float(input("5. Humidity (%) [10-100]: "))
        ph = float(input("6. pH Level [4.0-9.0]: "))
        rain = float(input("7. Rainfall (mm) [20-300]: "))
        
        # --- SECONDARY INPUTS (No longer defaults!) ---
        print("\n--- Advanced Features ---")
        oc = float(input("8. Organic Carbon [0.1 - 1.0]: "))
        soil_moisture = float(input("9. Soil Moisture (%) [10 - 90]: "))
        wind = float(input("10. Wind Speed (m/s) [1.0 - 10.0]: "))
        solar = float(input("11. Solar Radiation (W/m2) [150 - 350]: "))
        evap = float(input("12. Evapotranspiration (mm) [2.0 - 10.0]: "))

    except ValueError:
        print("\n❌ Invalid Input! Please enter numeric values only.")
        return

    # Prepare Data
    input_data = [n, p, k, temp, humid, ph, rain, oc, soil_moisture, wind, solar, evap]
    features_df = pd.DataFrame([input_data], columns=feature_names)

    print("\n⏳ Analyzing Data...")

    # --- TOP 3 LOGIC ---
    probabilities = model.predict_proba(features_df)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = encoder.inverse_transform(top_3_indices)
    top_3_scores = probabilities[top_3_indices]

    # --- DISPLAY RESULTS ---
    print("\n" + "="*45)
    print("      🌱 TOP 3 CROP RECOMMENDATIONS 🌱")
    print("="*45)

    for i in range(3):
        crop = top_3_crops[i]
        score = top_3_scores[i] * 100
        
        if i == 0:
            print(f"🥇 1. {crop.upper()} \t(Confidence: {score:.2f}%)")
        elif i == 1:
            print(f"🥈 2. {crop} \t(Confidence: {score:.2f}%)")
        else:
            print(f"🥉 3. {crop} \t(Confidence: {score:.2f}%)")
    
    print("="*45)

if __name__ == "__main__":
    predict_crop()