import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from groq import Groq

# =====================================================
# PATHS
# =====================================================
INPUT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/data/processed/final_enriched_data.csv"
OUTPUT_TEXT_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/outputs/llm_summary.txt"
OUTPUT_JSON_PATH = "/Users/venkatsaisubashpanchakarla/Desktop/Clinical_Trials/Clinical_Trials/Clinical_Trials/outputs/llm_summary.json"

# =====================================================
# GROQ CONFIG
# =====================================================
# Set in terminal:
# export GROQ_API_KEY="your_groq_api_key"
MODEL_NAME = "llama-3.1-8b-instant"

# =====================================================
# LOAD DATA
# =====================================================
def load_data() -> pd.DataFrame:
    path = Path(INPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(path)
    print(f"Loaded final enriched dataset: {df.shape}")
    return df


# =====================================================
# GET LATEST VISIT PER PATIENT
# =====================================================
def get_latest_visits(df: pd.DataFrame) -> pd.DataFrame:
    if "is_latest_visit" in df.columns:
        latest_df = df[df["is_latest_visit"] == 1].copy()
        if len(latest_df) > 0:
            return latest_df

    latest_df = (
        df.sort_values(["patient_id", "visit_number"])
        .groupby("patient_id", as_index=False)
        .tail(1)
        .copy()
    )
    return latest_df


# =====================================================
# COMPUTE SUMMARY METRICS
# =====================================================
def compute_summary_metrics(df: pd.DataFrame) -> dict:
    latest_df = get_latest_visits(df)

    total_records = int(len(df))
    total_patients = int(latest_df["patient_id"].nunique())

    high_risk_patients = 0
    if "predicted_risk_band" in latest_df.columns:
        high_risk_patients = int((latest_df["predicted_risk_band"] == "High").sum())

    medium_risk_patients = 0
    if "predicted_risk_band" in latest_df.columns:
        medium_risk_patients = int((latest_df["predicted_risk_band"] == "Medium").sum())

    low_risk_patients = 0
    if "predicted_risk_band" in latest_df.columns:
        low_risk_patients = int((latest_df["predicted_risk_band"] == "Low").sum())

    anomaly_patients = 0
    if "anomaly_flag" in latest_df.columns:
        anomaly_patients = int(latest_df["anomaly_flag"].sum())

    avg_temp = round(float(latest_df["temperature_f"].mean()), 2) if "temperature_f" in latest_df.columns else None
    avg_spo2 = round(float(latest_df["spo2_pct"].mean()), 2) if "spo2_pct" in latest_df.columns else None
    avg_hr = round(float(latest_df["heart_rate_bpm"].mean()), 2) if "heart_rate_bpm" in latest_df.columns else None
    avg_rr = round(float(latest_df["respiratory_rate_bpm"].mean()), 2) if "respiratory_rate_bpm" in latest_df.columns else None
    avg_adherence = round(float(latest_df["medication_adherence_pct"].mean()), 2) if "medication_adherence_pct" in latest_df.columns else None
    avg_wellbeing = round(float(latest_df["wellbeing_score_0_100"].mean()), 2) if "wellbeing_score_0_100" in latest_df.columns else None

    treatment_arm_summary = {}
    if "study_arm" in latest_df.columns:
        agg_map = {
            "patients": ("patient_id", "count"),
        }

        if "temperature_f" in latest_df.columns:
            agg_map["avg_temp_f"] = ("temperature_f", "mean")
        if "spo2_pct" in latest_df.columns:
            agg_map["avg_spo2_pct"] = ("spo2_pct", "mean")
        if "medication_adherence_pct" in latest_df.columns:
            agg_map["avg_adherence_pct"] = ("medication_adherence_pct", "mean")
        if "anomaly_flag" in latest_df.columns:
            agg_map["anomaly_count"] = ("anomaly_flag", "sum")
        if "predicted_risk_band" in latest_df.columns:
            # add risk counts per arm manually below
            pass

        grouped = latest_df.groupby("study_arm").agg(**agg_map).round(2)

        for arm_name, row in grouped.iterrows():
            arm_df = latest_df[latest_df["study_arm"] == arm_name]
            treatment_arm_summary[arm_name] = row.to_dict()

            if "predicted_risk_band" in arm_df.columns:
                treatment_arm_summary[arm_name]["high_risk_count"] = int((arm_df["predicted_risk_band"] == "High").sum())
                treatment_arm_summary[arm_name]["medium_risk_count"] = int((arm_df["predicted_risk_band"] == "Medium").sum())
                treatment_arm_summary[arm_name]["low_risk_count"] = int((arm_df["predicted_risk_band"] == "Low").sum())

    top_high_risk_patients = []
    if "predicted_adverse_event_probability" in latest_df.columns:
        cols = [
            "patient_id",
            "visit_number",
            "study_arm",
            "temperature_f",
            "spo2_pct",
            "heart_rate_bpm",
            "crp_mg_L",
            "symptom_burden",
            "predicted_adverse_event_probability",
            "predicted_risk_band",
            "anomaly_flag",
            "recommended_action",
        ]
        cols = [c for c in cols if c in latest_df.columns]

        top_high_risk_patients = (
            latest_df.sort_values("predicted_adverse_event_probability", ascending=False)[cols]
            .head(5)
            .to_dict(orient="records")
        )

    top_anomalous_patients = []
    if "anomaly_score" in latest_df.columns:
        cols = [
            "patient_id",
            "visit_number",
            "study_arm",
            "temperature_f",
            "spo2_pct",
            "heart_rate_bpm",
            "crp_mg_L",
            "symptom_burden",
            "anomaly_score",
            "anomaly_level",
            "predicted_risk_band",
            "recommended_action",
        ]
        cols = [c for c in cols if c in latest_df.columns]

        top_anomalous_patients = (
            latest_df.sort_values("anomaly_score", ascending=False)[cols]
            .head(5)
            .to_dict(orient="records")
        )

    common_recommended_actions = {}
    if "recommended_action" in latest_df.columns:
        common_recommended_actions = {
            str(k): int(v)
            for k, v in latest_df["recommended_action"].value_counts().head(5).to_dict().items()
        }

    critical_alert_counts = {}
    alert_cols = [
        "critical_spo2_alert",
        "high_fever_alert",
        "high_crp_alert",
        "tachycardia_alert",
        "respiratory_distress_alert",
        "worsening_trend_alert",
    ]
    for col in alert_cols:
        if col in latest_df.columns:
            critical_alert_counts[col] = int(latest_df[col].sum())

    metrics = {
        "total_records": total_records,
        "total_patients": total_patients,
        "high_risk_patients": high_risk_patients,
        "medium_risk_patients": medium_risk_patients,
        "low_risk_patients": low_risk_patients,
        "anomaly_patients": anomaly_patients,
        "average_temperature_f": avg_temp,
        "average_spo2_pct": avg_spo2,
        "average_heart_rate_bpm": avg_hr,
        "average_respiratory_rate_bpm": avg_rr,
        "average_medication_adherence_pct": avg_adherence,
        "average_wellbeing_score": avg_wellbeing,
        "critical_alert_counts": critical_alert_counts,
        "treatment_arm_summary": treatment_arm_summary,
        "top_high_risk_patients": top_high_risk_patients,
        "top_anomalous_patients": top_anomalous_patients,
        "common_recommended_actions": common_recommended_actions,
    }

    return metrics


# =====================================================
# BUILD PROMPT
# =====================================================
def build_prompt(metrics: dict) -> str:
    return f"""
You are a clinical trial safety analytics assistant.

Using the structured metrics below, generate a concise and professional summary for a hackathon demo dashboard.

Return the response in exactly these sections:

Executive Summary:
- 5 to 7 lines maximum

Top 3 Critical Safety Insights:
- bullet 1
- bullet 2
- bullet 3

Top 3 Recommended Actions:
- bullet 1
- bullet 2
- bullet 3

Treatment Arm Comparison:
- 2 to 4 lines maximum

Rules:
- Do not invent facts
- Use only the data provided
- Focus on patient safety, risk prioritization, anomalies, and next actions
- Keep the language simple, sharp, and presentation-ready

Structured metrics:
{json.dumps(metrics, indent=2)}
""".strip()


# =====================================================
# GENERATE SUMMARY WITH GROQ
# =====================================================
def generate_llm_summary(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in your environment.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You generate concise, factual clinical trial dashboard summaries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=700,
    )

    return response.choices[0].message.content.strip()


# =====================================================
# SAVE OUTPUTS
# =====================================================
def save_outputs(summary_text: str, metrics: dict) -> None:
    text_path = Path(OUTPUT_TEXT_PATH)
    json_path = Path(OUTPUT_JSON_PATH)

    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    text_path.write_text(summary_text, encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Summary text saved to: {OUTPUT_TEXT_PATH}")
    print(f"Summary metrics saved to: {OUTPUT_JSON_PATH}")


# =====================================================
# MAIN
# =====================================================
def main():
    df = load_data()
    metrics = compute_summary_metrics(df)
    prompt = build_prompt(metrics)

    print("Sending metrics to Groq...")
    summary_text = generate_llm_summary(prompt)

    save_outputs(summary_text, metrics)

    print("\n========== LLM SUMMARY ==========\n")
    print(summary_text)
    print("\n=================================\n")


if __name__ == "__main__":
    main()