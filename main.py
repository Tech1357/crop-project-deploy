import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("="*60)
    print("🚀 TRAINING ON CORRECTED AGRICULTURAL DATA")
    print("="*60)

    # 1. LOAD THE FIXED DATA
    # Note: Ensure you ran fix_dataset.py first!
    filename = "corrected_crop_dataset.csv" 
    try:
        df = pd.read_csv(filename)
        print(f"Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'corrected_crop_dataset.csv' not found.")
        print("   Run 'fix_dataset.py' first!")
        return

    # 2. PREPARE DATA
    # We can drop District/State now because the soil data is actually correct!
    # The model will learn from N, P, K, Rain, etc.
    
    feature_cols = [
        'N', 'P', 'K', 'pH', 'organic_carbon', 'soil_moisture',
        'temperature_c', 'humidity_pct', 'rainfall_mm'
    ]
    
    X = df[feature_cols]
    y = df['crop']

    # 3. ENCODE TARGET
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # 5. TRAIN RANDOM FOREST
    print(f"\nTraining Random Forest on {len(X_train)} samples...")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    # 6. EVALUATION
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*40)
    print(f"✅ FINAL TEST ACCURACY: {acc * 100:.2f}%")
    print("="*40)

    # Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. SAVE
    joblib.dump(model, 'crop_model_final.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("\nModel saved as 'crop_model_final.pkl'")

if __name__ == "__main__":
    main()