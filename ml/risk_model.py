import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =====================================================
# PATHS
# =====================================================
INPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/processed_data.csv"
OUTPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/predicted_data.csv"
MODEL_PATH = "models/risk_model.pkl"

# =====================================================
# TARGET + FEATURES
# =====================================================
TARGET_COLUMN = "adverse_event_flag"

FEATURE_COLUMNS = [
    "age",
    "gender",
    "study_arm",
    "visit_number",
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
    "urine_specific_gravity",
    "blood_test_abnormal",
    "urine_test_abnormal",
    "missed_visits_since_last",
    "response_delay_hours",
    "medication_adherence_pct",
    "wellbeing_score_0_100",
    "fever_severity_0_5",
    "cough_severity_0_5",
    "fatigue_severity_0_5",
    "shortness_of_breath_severity_0_5",
    "headache_severity_0_5",
    "myalgia_severity_0_5",
    "symptom_burden",
    "abnormal_score",
    "temp_change",
    "spo2_change",
    "worsening_temp",
    "spo2_drop",
    "risk_score",
]

NUMERIC_FEATURES = [
    "age",
    "visit_number",
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
    "urine_specific_gravity",
    "blood_test_abnormal",
    "urine_test_abnormal",
    "missed_visits_since_last",
    "response_delay_hours",
    "medication_adherence_pct",
    "wellbeing_score_0_100",
    "fever_severity_0_5",
    "cough_severity_0_5",
    "fatigue_severity_0_5",
    "shortness_of_breath_severity_0_5",
    "headache_severity_0_5",
    "myalgia_severity_0_5",
    "symptom_burden",
    "abnormal_score",
    "temp_change",
    "spo2_change",
    "worsening_temp",
    "spo2_drop",
    "risk_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "study_arm",
]


def load_data() -> pd.DataFrame:
    path = Path(INPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {INPUT_PATH}")

    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processed file: {missing}")


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    print("\nModel Evaluation")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    if y_prob is not None:
        print("\nSample predicted probabilities:")
        print(y_prob[:10])

    return pipeline


def add_predictions(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    df = df.copy()
    X_all = df[FEATURE_COLUMNS]

    df["predicted_adverse_event_flag"] = pipeline.predict(X_all)
    df["predicted_adverse_event_probability"] = pipeline.predict_proba(X_all)[:, 1].round(4)

    def map_risk_band(prob: float) -> str:
        if prob >= 0.70:
            return "High"
        elif prob >= 0.40:
            return "Medium"
        return "Low"

    df["predicted_risk_band"] = df["predicted_adverse_event_probability"].apply(map_risk_band)
    return df


def show_feature_importance(pipeline: Pipeline) -> None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        print("\nTop 15 Feature Importances")
        print(importance_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"Could not show feature importances: {e}")


def save_outputs(df: pd.DataFrame, pipeline: Pipeline) -> None:
    output_path = Path(OUTPUT_PATH)
    model_path = Path(MODEL_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    joblib.dump(pipeline, model_path)

    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print(f"Saved model to: {MODEL_PATH}")


def main():
    df = load_data()
    validate_columns(df)

    pipeline = train_model(df)
    df_with_predictions = add_predictions(df, pipeline)
    save_outputs(df_with_predictions, pipeline)
    show_feature_importance(pipeline)

    print("\nDone. Preview:")
    preview_cols = [
        "patient_id",
        "visit_number",
        "adverse_event_flag",
        "predicted_adverse_event_flag",
        "predicted_adverse_event_probability",
        "predicted_risk_band",
    ]
    existing_preview_cols = [col for col in preview_cols if col in df_with_predictions.columns]
    print(df_with_predictions[existing_preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()