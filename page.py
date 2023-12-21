import streamlit as st
import pandas as pd


def streamlitShow(X_train, y_train, logistic_regression, random_forest_classifier):
    st.set_page_config(page_title="Bank Loan Acceptance Prediction")
 

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio("Select a Model", ("Logistic Regression", "Random Forest Classifier"))

    
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=18)
    Experience = st.sidebar.number_input("Experience", min_value=-1, max_value=50, value=25)
    Income = st.sidebar.number_input("Income", min_value=0, max_value=500, value=100)
    ZIPcode = st.sidebar.number_input("ZIP.Code", min_value=90000, max_value=100000, value=90000)
    Family = st.sidebar.number_input("Family", min_value=0, max_value=20, value=0)
    CCAvg = st.sidebar.number_input("CCavg", min_value=0.1, max_value=10.0, value=5.0)
    Education = st.sidebar.number_input("Education", min_value=0, max_value=3, value=2)
    Mortgage = st.sidebar.number_input("Mortgage", min_value=0, max_value=1000, value=0)
    Securities_Acc = st.sidebar.number_input("Securities Account?", min_value=0, max_value=1, value=0)
    CD_Acc = st.sidebar.number_input("CD Account?", min_value=0, max_value=1, value=0)
    Online = st.sidebar.number_input("Online?", min_value=0, max_value=1, value=0)
    CreditCard = st.sidebar.number_input("Credit Card?", min_value=0, max_value=1, value=0)


    

    

    if st.sidebar.button("Predict"):
        if model_choice == "Logistic Regression":
            model = logistic_regression
        else:
            model = random_forest_classifier

        # Data preparation
        user_input_data = pd.DataFrame.from_dict(feature_values, orient="index").transpose()
        user_input_data = user_input_data.astype({"Age": int, "Experience": int, "Income": int, "ZIP.Code": int, "Family": int, "CCAvg": float,"Education" : int,"Mortgage":int,"Securities.Account":int,"CD.Account":int,"Online":int,"CreditCard":int})
      
        model.fit(X_train, y_train)

        prediction = model.predict(user_input_data)

        # Prediction probabilities
        prediction_probabilities = model.predict_proba(user_input_data)

        # Display the predicted value as text
        st.markdown("<h2 style='color:white; font-size: 30px; font-weight: normal; text-align: center;'>Based on the provided features, you are predicted to: </h2>", unsafe_allow_html=True)
        if prediction[0] == 1:
            set_color_text('lime', 'Be accepted for a bank loan:)')
        else:
            set_color_text('red', 'Not be accepted for a bank loan:(')

        st.markdown("<h3 style='color:white; font-size: 30px; font-weight: normal; text-align: center;'>Prediction Probability (0: No, 1: Yes): </h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:white; font-size: 24px; font-weight: normal; text-align: center;'>0: {prediction_probabilities[0][0]:.5f}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:white; font-size: 24px; font-weight: normal; text-align: center;'>1: {prediction_probabilities[0][1]:.5f}</h3>", unsafe_allow_html=True)

