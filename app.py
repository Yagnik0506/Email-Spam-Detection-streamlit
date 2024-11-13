import streamlit as st
import pickle

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("This is a Machine Learning application to classify emails as spam or ham.")
user_input =st.text_area("Enter an Email to Classify",height = 150)
if st.button("Classify"):
    if user_input:
        data = [user_input]
        vact = cv.transform(data).toarray()
        pred = model.predict(vact)
        if pred[0]==0:
            st.success("This Emailo is Not Spam")
        else:
            st.error("This Email is Spam")
    else:
        print("Please type Email")
        

