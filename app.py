import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = False
    st.session_state["best_model"] = None
    st.session_state["scaler"] = None
    st.session_state["best_model_name"] = None

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Earnings Manipulation Detector",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM THEME (YELLOW–BLACK)
# ---------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0f0f0f; color: #f5c518; }
[data-testid="stSidebar"] { background-color: #1a1a1a; }
h1, h2, h3, h4 { color: #f5c518; }

.stButton > button {
    background-color: #f5c518;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #ffd84d;
}

[data-testid="metric-container"] {
    background-color: #1f1f1f;
    border-left: 5px solid #f5c518;
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
MODELS = {
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
}

# ---------------------------------------------------
# PARAMETER GRIDS
# ---------------------------------------------------
PARAM_GRIDS = {
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },
    "Naive Bayes": {
        "var_smoothing": np.logspace(0, -9, 5)
    },
    "AdaBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4]
    }
}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def needs_scaling(model_name):
    return model_name in ["SVM", "KNN"]

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("📊 Earnings Manipulation Classification Dashboard")

st.markdown("""
Upload a **financial dataset** to detect earnings manipulation.

**Required columns:**  
`DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `ACCR`, `LEVI`, `Manipulator`
""")

uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        required_cols = [
            "DSRI", "GMI", "AQI", "SGI", "DEPI",
            "SGAI", "ACCR", "LEVI", "Manipulator"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ Missing required columns")
            st.stop()

        X = df[required_cols[:-1]]
        y = df["Manipulator"].map({"No": 0, "Yes": 1})

        if y.isnull().any():
            st.error("❌ Manipulator column must contain only Yes / No")
            st.stop()

        # Sidebar
        st.sidebar.header("⚙️ Settings")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25, step=0.05)
        use_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning")
        run = st.sidebar.button("🚀 Run All Models")

        if run:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            results = []

            with st.spinner("Training models..."):
                for model_name, model in MODELS.items():

                    X_tr, X_te = X_train, X_test

                    if needs_scaling(model_name):
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_train)
                        X_te = scaler.transform(X_test)
                    else:
                        scaler = None

                    if use_tuning:
                        grid = GridSearchCV(
                            model,
                            PARAM_GRIDS[model_name],
                            cv=5,
                            scoring="roc_auc",
                            n_jobs=-1
                        )
                        grid.fit(X_tr, y_train)
                        best_estimator = grid.best_estimator_
                    else:
                        model.fit(X_tr, y_train)
                        best_estimator = model

                    y_pred = best_estimator.predict(X_te)
                    y_prob = best_estimator.predict_proba(X_te)[:, 1]

                    metrics = evaluate(y_test, y_pred, y_prob)

                    results.append({
                        "Model": model_name,
                        **metrics
                    })

                    # Save the best estimator temporarily for final retrain
                    if model_name == list(MODELS.keys())[0]:
                        temp_models = {}
                    temp_models[model_name] = (best_estimator, scaler)

            # Results table
            results_df = pd.DataFrame(results).round(4)

            st.subheader("📊 Model Comparison Results")
            st.dataframe(results_df)

            # --- CHOOSE BEST MODEL: Prioritize Recall, then Accuracy ---
            best_model_row = results_df.sort_values(
                by=["Recall", "Accuracy"], ascending=False
            ).iloc[0]
            best_model_name = best_model_row["Model"]

            # Retrain best model on full training set
            best_model_obj, scaler = temp_models[best_model_name]
            if needs_scaling(best_model_name):
                scaler_full = StandardScaler()
                X_train_scaled = scaler_full.fit_transform(X_train)
                best_model_obj.fit(X_train_scaled, y_train)
                st.session_state["scaler"] = scaler_full
            else:
                best_model_obj.fit(X_train, y_train)
                st.session_state["scaler"] = None

            st.session_state["best_model"] = best_model_obj
            st.session_state["best_model_name"] = best_model_name
            st.session_state["model_ready"] = True

            st.success(
                f"""
🏆 **Best Performing Algorithm** (Prioritizing Recall)

• **Model:** {best_model_name}  
• **ROC-AUC:** {best_model_row['ROC-AUC']}  
• **F1-score:** {best_model_row['F1-score']}  
• **Recall:** {best_model_row['Recall']}  
• **Accuracy:** {best_model_row['Accuracy']}
"""
            )

            # Beneish baseline
            st.markdown("---")
            st.subheader("📉 Beneish M-Score Baseline")
            st.dataframe(pd.DataFrame([{
                "Accuracy": 0.8364,
                "Precision": 0.5556,
                "Recall": 0.5000,
                "F1-score": 0.5263,
                "ROC-AUC": 0.9044
            }]).round(4))

            # --- USER INPUT FOR PREDICTION ---
            st.markdown("---")
            st.subheader("🔮 Predict Manipulation for New Observation")

            col1, col2, col3, col4 = st.columns(4)
            inputs = {}
            with col1:
                inputs["DSRI"] = st.number_input("DSRI", value=1.0, format="%.4f")
                inputs["DEPI"] = st.number_input("DEPI", value=1.0, format="%.4f")
            with col2:
                inputs["GMI"] = st.number_input("GMI", value=1.0, format="%.4f")
                inputs["SGAI"] = st.number_input("SGAI", value=1.0, format="%.4f")
            with col3:
                inputs["AQI"] = st.number_input("AQI", value=1.0, format="%.4f")
                inputs["ACCR"] = st.number_input("ACCR", value=0.0, format="%.4f")
            with col4:
                inputs["SGI"] = st.number_input("SGI", value=1.0, format="%.4f")
                inputs["LEVI"] = st.number_input("LEVI", value=1.0, format="%.4f")

            user_X = pd.DataFrame([inputs])

            # Apply scaling if needed
            if st.session_state["scaler"]:
                user_X_scaled = st.session_state["scaler"].transform(user_X)
            else:
                user_X_scaled = user_X

            # Predict
            model = st.session_state["best_model"]
            pred = model.predict(user_X_scaled)[0]
            prob = model.predict_proba(user_X_scaled)[0, 1]

            st.markdown("### 🧾 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ **Earnings Manipulation Likely** (Probability: {prob:.2%})")
            else:
                st.success(f"✅ **No Manipulation Detected** (Probability of Manipulation: {prob:.2%})")

            st.info(f"Model used: **{st.session_state['best_model_name']}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload an Excel file to start analysis.")
