import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =====================================================
# PATHS
# =====================================================
INPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/predicted_data.csv"
OUTPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/final_enriched_data.csv"

# =====================================================
# FEATURES FOR ANOMALY DETECTION
# =====================================================
ANOMALY_FEATURES = [
    "temperature_f",
    "heart_rate_bpm",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "spo2_pct",
    "respiratory_rate_bpm",
    "wbc_k_per_uL",
    "crp_mg_L",
    "alt_u_L",
    "creatinine_mg_dL",
    "symptom_burden",
    "abnormal_score",
    "temp_change",
    "spo2_change",
    "risk_score",
    "predicted_adverse_event_probability",
]

CONTAMINATION = 0.12


def load_data() -> pd.DataFrame:
    path = Path(INPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(path)
    print(f"✅ Loaded data for anomaly detection: {df.shape}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ANOMALY_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required anomaly feature columns: {missing}")


def prepare_features(df: pd.DataFrame):
    feature_df = df[ANOMALY_FEATURES].copy()

    # Fill any missing values safely
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    return X_scaled


def run_anomaly_detection(df: pd.DataFrame, X_scaled) -> pd.DataFrame:
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )

    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df = df.copy()

    # IsolationForest returns -1 for anomaly, 1 for normal
    df["anomaly_flag"] = pd.Series(preds).apply(lambda x: 1 if x == -1 else 0)

    # Lower score = more anomalous, so invert for readability
    df["anomaly_score"] = (-scores).round(4)

    def map_anomaly_level(score: float) -> str:
        if score >= 0.18:
            return "High"
        elif score >= 0.08:
            return "Medium"
        return "Low"

    df["anomaly_level"] = df["anomaly_score"].apply(map_anomaly_level)

    return df


def add_rule_based_alerts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["critical_spo2_alert"] = (df["spo2_pct"] <= 91).astype(int)
    df["high_fever_alert"] = (df["temperature_f"] >= 102.0).astype(int)
    df["high_crp_alert"] = (df["crp_mg_L"] >= 20).astype(int)
    df["tachycardia_alert"] = (df["heart_rate_bpm"] >= 110).astype(int)
    df["respiratory_distress_alert"] = (df["respiratory_rate_bpm"] >= 24).astype(int)
    df["worsening_trend_alert"] = (
        (df["temp_change"] >= 1.0) | (df["spo2_change"] <= -2.0)
    ).astype(int)

    df["total_alert_count"] = (
        df["critical_spo2_alert"]
        + df["high_fever_alert"]
        + df["high_crp_alert"]
        + df["tachycardia_alert"]
        + df["respiratory_distress_alert"]
        + df["worsening_trend_alert"]
        + df["anomaly_flag"]
    )

    return df


def add_recommended_action(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def recommend_action(row):
        if row["critical_spo2_alert"] == 1 or row["respiratory_distress_alert"] == 1:
            return "Immediate clinical review"
        elif row["high_fever_alert"] == 1 and row["high_crp_alert"] == 1:
            return "Urgent physician follow-up"
        elif row["anomaly_flag"] == 1 and row["predicted_risk_band"] == "High":
            return "Escalate for safety monitoring"
        elif row["predicted_risk_band"] == "High":
            return "Close monitoring required"
        elif row["anomaly_flag"] == 1:
            return "Review abnormal patient pattern"
        elif row["missed_visits_since_last"] >= 1:
            return "Contact patient for follow-up"
        else:
            return "Continue routine monitoring"

    df["recommended_action"] = df.apply(recommend_action, axis=1)
    return df


def save_data(df: pd.DataFrame) -> None:
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Final enriched dataset saved to: {OUTPUT_PATH}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n🔥 Anomaly Detection Summary")
    print(f"Total records: {len(df)}")
    print(f"Anomalies detected: {df['anomaly_flag'].sum()}")
    print("\nAnomaly level counts:")
    print(df["anomaly_level"].value_counts())

    print("\nTop anomalous records:")
    preview_cols = [
        "patient_id",
        "visit_number",
        "anomaly_flag",
        "anomaly_score",
        "anomaly_level",
        "predicted_risk_band",
        "recommended_action",
    ]
    existing_cols = [col for col in preview_cols if col in df.columns]
    print(
        df.sort_values("anomaly_score", ascending=False)[existing_cols]
        .head(10)
        .to_string(index=False)
    )


def main():
    df = load_data()
    validate_columns(df)
    X_scaled = prepare_features(df)
    df = run_anomaly_detection(df, X_scaled)
    df = add_rule_based_alerts(df)
    df = add_recommended_action(df)
    save_data(df)
    print_summary(df)


if __name__ == "__main__":
    main()