import streamlit as st
import joblib
import pandas as pd
import numpy as np

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")

st.sidebar.image(r"C:\ML\flag (1).jpg")
st.sidebar.title("About Project")
st.sidebar.write("Objective of this project is to predict sentiment (Neg/Pos) of food review")

st.sidebar.title("Libraries")
st.sidebar.markdown("- sklearn")
st.sidebar.markdown("- pandas")
st.sidebar.markdown("- numpy")

st.sidebar.title("Cloud")
st.sidebar.markdown("streamlit")

st.sidebar.title("Contact Us")
st.sidebar.write("Mob no:-6393376338")
st.sidebar.write("Email: Anuraggautam879@gmail.com")

st.markdown("""
<style>
.banner {
    background-image: url("https://images.unsplash.com/photo-1504674900247-0877df9cc836");
    background-size: cover;
    background-position: center;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 40px;
    font-weight: bold;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
}
</style>

<div class="banner">
    Food Sentiment Analysis
</div>
""", unsafe_allow_html=True)

st.write("\n")

col1,col2=st.columns([.4,.6])
with col1:
# Input box
        st.header("Predict Single Review")
        review = st.text_input("**Enter Review**")
        
        # Prediction button
        if st.button("Predict"):
        
            X_test = vectorizer.transform([review])
        
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
        
            if pred[0] == 0:
                st.error("**Sentiment = Negative 👎**")
                st.warning(f"Confidence Score: {prob[0][0]:.2f}")
        
            else:
                st.success("**Sentiment = Positive 👍**")
                st.warning(f"Confidence Score: {prob[0][1]:.2f}")

with col2:
        st.header("Predict Bulk Review")
        file=st.file_uploader("**Selec a csv file**",type=["csv","txt"])
        if file:
                df=pd.read_csv(file,header=None,names=["Review"])
                placeholder=st.empty()
                placeholder.dataframe(df)
                if st.button("Bulk predict"):
                        X_test = vectorizer.transform(df.Review)
                        pred=model.predict(X_test)
                        prob=model.predict_proba(X_test)
                        sentiment=["Positive" if i==1 else "Negative" for i in pred]
                        df["sentiment"]=sentiment
                        df["Confidence"]=np.max(prob,axis=1)
                        st.dataframe(df)
                