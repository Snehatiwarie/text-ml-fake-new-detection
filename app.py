import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Load trained model and vectorizer
# -----------------------------
model = pickle.load(open("fake_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection System")
st.write("Enter news text or upload a CSV file to classify news as Real or Fake.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About Project")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Feature Extraction: TF-IDF")
st.sidebar.write("Task: Fake News Classification")

# -----------------------------
# Manual text input
# -----------------------------
st.subheader("✍️ Check Single News Text")
text = st.text_area("Enter News Content")

if st.button("🔍 Predict News"):
    if text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        confidence = 0
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            confidence = max(probs) * 100

        if prediction == 1 or str(prediction).lower() == "fake":
            st.error(f"❌ Fake News\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Real News\n\nConfidence: {confidence:.2f}%")

        if probs is not None:
            st.subheader("📊 Prediction Confidence")
            prob_df = pd.DataFrame({
                "Class": ["Real", "Fake"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Class"))

# -----------------------------
# CSV upload for bulk testing
# -----------------------------
st.subheader("📂 Upload CSV Dataset for Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        df["text"] = df["text"].fillna("").astype(str)

        X_uploaded = vectorizer.transform(df["text"])
        predictions = model.predict(X_uploaded)

        df["prediction"] = predictions
        df["prediction_label"] = df["prediction"].replace({
            0: "Real",
            1: "Fake",
            "real": "Real",
            "fake": "Fake",
            "REAL": "Real",
            "FAKE": "Fake"
        })

        st.write("### Prediction Results")
        st.dataframe(df.head(10))

        st.subheader("📈 Prediction Distribution")
        pred_counts = df["prediction_label"].value_counts()

        if pred_counts.empty:
            st.warning("No prediction data available to plot.")
        else:
            fig1, ax1 = plt.subplots()
            pred_counts.plot(kind="bar", ax=ax1)
            ax1.set_title("Predicted Class Counts")
            ax1.set_xlabel("Class")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        if "label" in df.columns:
            st.subheader("🎯 Accuracy on Uploaded File")

            true_labels = df["label"]

            if true_labels.dtype == object:
                true_labels = true_labels.astype(str).str.lower().replace({
                    "real": 0,
                    "fake": 1
                })

            pred_numeric = df["prediction"].replace({
                "real": 0,
                "fake": 1,
                "REAL": 0,
                "FAKE": 1
            })

            acc = accuracy_score(true_labels, pred_numeric)
            st.success(f"Accuracy: {acc*100:.2f}%")

            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(true_labels, pred_numeric)

            fig2, ax2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            ax2.set_title("Confusion Matrix")
            st.pyplot(fig2)