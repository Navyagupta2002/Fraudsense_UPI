import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------
# FIX RANDOMNESS (Optional - stable accuracy)
# -------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(page_title="FraudSense UPI", layout="wide")

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("🔐 FraudSense UPI")
page = st.sidebar.radio("Navigation", ["Home", "Dashboard", "Fraud Detection"])

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Upi fraud dataset final.csv")

try:
    df = load_data()
except:
    st.error("CSV file not found. Keep CSV in same folder as app.py")
    st.stop()

# -------------------------------------------------
# COMMON PREPROCESSING (SAFE)
# -------------------------------------------------

# Safe datetime extraction
if "Date" in df.columns and "Time" in df.columns:
    df["Transaction_DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        errors="coerce"
    )
    df["hour"] = df["Transaction_DateTime"].dt.hour
    df["day_of_week"] = df["Transaction_DateTime"].dt.day_name()
    df["month"] = df["Transaction_DateTime"].dt.month_name()

# Drop HIGH CARDINALITY columns (IMPORTANT FIX)
drop_cols = [
    "Transaction_ID",
    "Customer_ID",
    "Merchant_ID",
    "Device_ID",
    "IP_Address",
    "Date",
    "Time",
    "Transaction_DateTime"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode only selected categorical columns
categorical_cols = [
    "Transaction_Type",
    "Payment_Gateway",
    "Transaction_City",
    "Transaction_State",
    "Transaction_Status",
    "Device_OS",
    "Merchant_Category",
    "Transaction_Channel",
    "day_of_week",
    "month"
]

categorical_cols = [c for c in categorical_cols if c in df.columns]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Check target
if "fraud" not in df.columns:
    st.error("Column 'fraud' not found in dataset.")
    st.write(df.columns)
    st.stop()

# Scale numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("fraud")

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop("fraud", axis=1)
y = df["fraud"]

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if page == "Home":

    st.title("🔐 FraudSense UPI")
    st.subheader("AI-Powered UPI Fraud Detection System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{y.sum():,}")
    col3.metric("Fraud Rate", f"{y.mean()*100:.2f}%")

# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------
elif page == "Dashboard":

    st.title("📊 Fraud Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Distribution")
        fig1, ax1 = plt.subplots()
        y.value_counts().plot(kind="bar")
        st.pyplot(fig1)

    with col2:
        st.subheader("Fraud Percentage")
        fig2, ax2 = plt.subplots()
        y.value_counts().plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(fig2)

# -------------------------------------------------
# FRAUD DETECTION PAGE
# -------------------------------------------------
elif page == "Fraud Detection":

    st.title("🧠 ANN Fraud Detection Model")

    # Sidebar Hyperparameters
    st.sidebar.header("Model Controls")

    hidden1 = st.sidebar.slider("Hidden Layer 1 Units", 64, 512, 256)
    hidden2 = st.sidebar.slider("Hidden Layer 2 Units", 32, 256, 128)
    hidden3 = st.sidebar.slider("Hidden Layer 3 Units", 16, 128, 64)

    dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=1)
    epochs = st.sidebar.slider("Epochs", 10, 60, 30)

    if st.button("🚀 Train Model"):

        with st.spinner("Training model..."):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )

            class_weight_dict = dict(enumerate(class_weights))

            model = Sequential([
                Dense(hidden1, activation='relu', input_shape=(X_train.shape[1],)),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(hidden2, activation='relu'),
                BatchNormalization(),
                Dropout(dropout_rate),

                Dense(hidden3, activation='relu'),
                BatchNormalization(),
                Dropout(dropout_rate/2),

                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
            )

            early_stop = EarlyStopping(
                monitor="val_auc",
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=64,
                class_weight=class_weight_dict,
                callbacks=[early_stop],
                verbose=0
            )

            results = model.evaluate(X_test, y_test, verbose=0)

        st.success("Model Trained Successfully ✅")

        accuracy = results[1]
        auc_score = results[2]

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("AUC Score", f"{auc_score:.3f}")

        # ROC Curve
        y_pred_prob = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        st.subheader("ROC Curve")
        fig3, ax3 = plt.subplots()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()
        st.pyplot(fig3)

        # Confusion Matrix
        y_pred = (y_pred_prob > 0.5).astype("int32")
        st.subheader("Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
        st.pyplot(fig4)