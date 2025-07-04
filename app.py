import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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
