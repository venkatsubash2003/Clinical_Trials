import pandas as pd
import numpy as np
from pathlib import Path

# 🔥 Hardcoded paths (EDIT THIS ONLY IF NEEDED)
INPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/raw/clinical_trial_safety_dataset_100rows.csv"
OUTPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/processed_data.csv"


SYMPTOM_COLUMNS = [
    "fever_severity_0_5",
    "cough_severity_0_5",
    "fatigue_severity_0_5",
    "shortness_of_breath_severity_0_5",
    "headache_severity_0_5",
    "myalgia_severity_0_5",
]


def load_data():
    path = Path(INPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(path)
    print("✅ Data loaded:", df.shape)
    return df


def basic_cleaning(df):
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df = df.sort_values(["patient_id", "visit_number"])

    # Fill missing values
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def create_base_features(df):
    # Symptom score
    df["symptom_burden"] = df[SYMPTOM_COLUMNS].mean(axis=1)

    # Vitals flags
    df["fever_flag"] = (df["temperature_f"] > 100.4).astype(int)
    df["low_spo2_flag"] = (df["spo2_pct"] < 94).astype(int)
    df["tachycardia_flag"] = (df["heart_rate_bpm"] > 100).astype(int)

    # Abnormal count
    df["abnormal_score"] = (
        df["fever_flag"]
        + df["low_spo2_flag"]
        + df["tachycardia_flag"]
        + df["blood_test_abnormal"]
        + df["urine_test_abnormal"]
    )

    return df


def create_time_features(df):
    df = df.copy()

    # Previous visit
    df["prev_temp"] = df.groupby("patient_id")["temperature_f"].shift(1)
    df["prev_spo2"] = df.groupby("patient_id")["spo2_pct"].shift(1)

    # Change
    df["temp_change"] = df["temperature_f"] - df["prev_temp"]
    df["spo2_change"] = df["spo2_pct"] - df["prev_spo2"]

    # Trends
    df["worsening_temp"] = (df["temp_change"] > 1).astype(int)
    df["spo2_drop"] = (df["spo2_change"] < -2).astype(int)

    # Fill first visit nulls
    df = df.fillna(0)

    return df


def create_risk_score(df):
    df = df.copy()

    df["risk_score"] = (
        df["abnormal_score"]
        + df["worsening_temp"]
        + df["spo2_drop"] * 2
        + df["missed_visits_since_last"]
        + (df["medication_adherence_pct"] < 80).astype(int)
        + df["adverse_event_flag"] * 2
    )

    # Risk levels
    df["risk_level"] = df["risk_score"].apply(
        lambda x: "High" if x >= 6 else "Medium" if x >= 3 else "Low"
    )

    return df


def save_data(df):
    path = Path(OUTPUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to: {OUTPUT_PATH}")


def main():
    df = load_data()
    df = basic_cleaning(df)
    df = create_base_features(df)
    df = create_time_features(df)
    df = create_risk_score(df)
    save_data(df)

    print("\n🔥 Preprocessing complete!")
    print(df[["patient_id", "visit_number", "risk_score", "risk_level"]].head())


if __name__ == "__main__":
    main()