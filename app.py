"""
Employee Attrition Analysis and Prediction - Streamlit App

Run:
    streamlit run app.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocessing import full_pipeline  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee_attrition.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Employee Attrition Analysis & Prediction",
    page_icon="📊",
    layout="wide",
)


# ---------- Cached loaders ----------
@st.cache_data
def load_data():
    return full_pipeline(DATA_PATH)


@st.cache_resource
def load_model():
    bundle = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    return bundle


@st.cache_data
def load_comparison():
    with open(os.path.join(MODELS_DIR, "model_comparison.json")) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_at_risk():
    return pd.read_csv(os.path.join(MODELS_DIR, "at_risk_employees.csv"))


df = load_data()
model_bundle = load_model()
pipeline = model_bundle["pipeline"]
comparison = load_comparison()
feature_importance = load_feature_importance()
at_risk_df = load_at_risk()

st.title("📊 Employee Attrition Analysis & Prediction")
st.caption(
    "HR analytics dashboard: explore attrition drivers, review model performance, "
    "and identify employees at risk of leaving."
)

tab_overview, tab_eda, tab_model, tab_predict, tab_at_risk = st.tabs(
    ["Overview", "Explore the Data", "Model Performance", "Predict Attrition", "At-Risk Employees"]
)

# ---------------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    total = len(df)
    left = int(df["Attrition"].sum())
    rate = left / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{total:,}")
    c2.metric("Employees Who Left", f"{left:,}")
    c3.metric("Attrition Rate", f"{rate:.1f}%")
    c4.metric("Best Model AUC-ROC", f"{comparison['results'][comparison['best_model']]['auc_roc']:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        dept = (
            df.groupby("Department")["Attrition"]
            .agg(["count", "sum"])
            .rename(columns={"count": "Total", "sum": "Left"})
        )
        dept["Rate (%)"] = (dept["Left"] / dept["Total"] * 100).round(1)
        fig = px.bar(dept.reset_index(), x="Department", y="Rate (%)", color="Department")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Attrition Overall Split")
        counts = df["Attrition"].map({0: "Stayed", 1: "Left"}).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This project analyzes employee data to identify the key drivers of attrition and "
        "build a predictive model that flags at-risk employees, so HR can act before they leave."
    )

# ---------------------------------------------------------------------------
# TAB 2: EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Explore What Drives Attrition")

    factor = st.selectbox(
        "Choose a factor to compare against Attrition",
        [
            "OverTime",
            "JobSatisfaction",
            "MonthlyIncome",
            "WorkLifeBalance",
            "DistanceFromHome",
            "YearsAtCompany",
            "BusinessTravel",
            "JobRole",
            "MaritalStatus",
            "Age",
            "TenureCategory",
            "SatisfactionScore",
        ],
    )

    plot_df = df.copy()
    plot_df["Attrition Status"] = plot_df["Attrition"].map({0: "Stayed", 1: "Left"})

    if df[factor].dtype == object or df[factor].nunique() <= 10:
        rate_df = (
            plot_df.groupby(factor)["Attrition"].mean().reset_index().rename(columns={"Attrition": "Attrition Rate"})
        )
        rate_df["Attrition Rate"] = (rate_df["Attrition Rate"] * 100).round(1)
        fig = px.bar(rate_df, x=factor, y="Attrition Rate", color=factor, title=f"Attrition Rate (%) by {factor}")
    else:
        fig = px.box(plot_df, x="Attrition Status", y=factor, color="Attrition Status", title=f"{factor} by Attrition Status")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    fig2 = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Model Performance
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader(f"Model Comparison (Best: {comparison['best_model']})")

    results = comparison["results"]
    metrics_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "auc_roc"]]
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metric Comparison")
        melted = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="index", y="Score", color="Metric", barmode="group")
        fig.update_layout(xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Confusion Matrix — {comparison['best_model']}")
        cm = results[comparison["best_model"]]["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual: Stayed", "Actual: Left"], columns=["Pred: Stayed", "Pred: Left"])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Features Driving Attrition")
    fi_plot = feature_importance.head(15).sort_values("importance")
    fig_fi = px.bar(fi_plot, x="importance", y="feature", orientation="h")
    st.plotly_chart(fig_fi, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict Attrition Risk for an Employee")
    st.caption("Fill in employee details to get a live attrition risk prediction.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 35)
            gender = st.selectbox("Gender", df["Gender"].unique())
            marital = st.selectbox("Marital Status", df["MaritalStatus"].unique())
            dept = st.selectbox("Department", df["Department"].unique())
            role = st.selectbox("Job Role", df["JobRole"].unique())
            edu_field = st.selectbox("Education Field", df["EducationField"].unique())
            education = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
        with c2:
            monthly_income = st.number_input("Monthly Income", 1000, 25000, 5000, step=100)
            job_level = st.slider("Job Level", 1, 5, 2)
            years_company = st.slider("Years at Company", 0, 40, 5)
            years_role = st.slider("Years in Current Role", 0, 20, 3)
            years_manager = st.slider("Years With Current Manager", 0, 20, 3)
            years_promo = st.slider("Years Since Last Promotion", 0, 20, 1)
            total_working_years = st.slider("Total Working Years", 0, 40, 8)
        with c3:
            overtime = st.selectbox("OverTime", df["OverTime"].unique())
            travel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
            distance = st.slider("Distance From Home (miles)", 1, 30, 5)
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            wlb = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            rel_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
            job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
            stock_option = st.slider("Stock Option Level (0-3)", 0, 3, 0)
            num_companies = st.slider("Number of Companies Worked", 0, 10, 2)
            training_times = st.slider("Training Times Last Year", 0, 6, 2)
            pct_hike = st.slider("Percent Salary Hike", 10, 25, 14)

        submitted = st.form_submit_button("Predict Attrition Risk", type="primary")

    if submitted:
        row = {
            "Age": age,
            "BusinessTravel": travel,
            "DailyRate": 800,
            "Department": dept,
            "DistanceFromHome": distance,
            "Education": education,
            "EducationField": edu_field,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": 65,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": 14000,
            "NumCompaniesWorked": num_companies,
            "OverTime": overtime,
            "PercentSalaryHike": pct_hike,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": rel_satisfaction,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": wlb,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager,
        }
        input_df = pd.DataFrame([row])

        # Re-use the same feature engineering as training
        from preprocessing import engineer_features

        input_df = engineer_features(input_df)

        proba = pipeline.predict_proba(input_df)[0, 1]
        pred = "Likely to Leave" if proba >= 0.5 else "Likely to Stay"

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Attrition Probability", f"{proba*100:.1f}%")
            if proba >= 0.6:
                st.error(f"⚠️ High Risk — {pred}")
            elif proba >= 0.35:
                st.warning(f"⚠️ Moderate Risk — {pred}")
            else:
                st.success(f"✅ Low Risk — {pred}")
        with col2:
            st.progress(min(proba, 1.0))
            st.caption(
                "This estimate is based on patterns learned from historical employee data. "
                "Use it as a decision-support signal, not a standalone HR decision."
            )

# ---------------------------------------------------------------------------
# TAB 5: At-Risk Employees
# ---------------------------------------------------------------------------
with tab_at_risk:
    st.subheader("Ranked List of At-Risk Employees")
    st.caption("Full workforce scored by the trained model, sorted by attrition probability (highest first).")

    threshold = st.slider("Minimum risk probability to show (%)", 0, 100, 40) / 100
    filtered = at_risk_df[at_risk_df["AttritionProbability"] >= threshold].copy()
    filtered["AttritionProbability"] = (filtered["AttritionProbability"] * 100).round(1)

    display_cols = [
        "Age", "Department", "JobRole", "MonthlyIncome", "OverTime",
        "JobSatisfaction", "YearsAtCompany", "AttritionProbability",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("AttritionProbability", ascending=False),
        use_container_width=True,
        height=500,
    )
    st.caption(f"Showing {len(filtered)} employees above the {int(threshold*100)}% risk threshold.")

    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download At-Risk List (CSV)", csv, "at_risk_employees.csv", "text/csv")
