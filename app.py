import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "message": [
        "Your account has been credited with 5000 Naira",
        "Your OTP is 45321 do not share it",
        "Click this link to verify your BVN immediately",
        "Congratulations you won 1 million Naira claim now",
        "Your transaction was successful",
        "Your account will be blocked update now"
    ],
    "label": [0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])

model = MultinomialNB()
model.fit(X, df["label"])

st.title("SMS Smishing Detection System")

sms_input = st.text_area("Enter SMS message:")

if st.button("Check Message"):
    input_vector = vectorizer.transform([sms_input])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        st.error("⚠️ SMISHING DETECTED")
    else:
        st.success("✅ LEGITIMATE SMS")
