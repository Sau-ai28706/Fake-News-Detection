
import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
    else:
        st.warning("Please enter some text to Analyze. ")

import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6J_Kp8d_Khh6a0MLHtN-Yf39SSYzWgBw17EC4KFauV4Fw")

for m in genai.list_models():
    print(m.name)

model_gemini = genai.GenerativeModel("gemini-3.5-flash")
response = model_gemini.generate_content(
    "Summarize the following news article in 4-5 bullet points:\n\n"
    + news_input
)

summary = response.text
st.subheader("News Summary")
st.write(summary)
