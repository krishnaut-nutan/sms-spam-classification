
import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("📩 SMS Spam Classifier")

# Text input
input_sms = st.text_area("Enter the message")

# Predict button
if st.button("Predict"):
    # Preprocess and predict
    transformed = vectorizer.transform([input_sms])
    result = model.predict(transformed)[0]

    # Show result
    st.subheader("Result:")
    st.success("✅ Not Spam") if result == 0 else st.error("🚨 Spam")
