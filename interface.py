import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
cancer_data = pd.read_csv("cancer.csv")

# Prepare features (X) and target (Y)
X = cancer_data.drop(columns='Diagnosis', axis=1)  # Exclude target column
Y = cancer_data['Diagnosis']  # Target variable

df = pd.DataFrame(cancer_data)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Streamlit Interface
st.title("Cancer Prediction System")
st.line_chart(df)

st.sidebar.title("Enter Input Values")
form = st.sidebar.form(key='cancer_form')

# Input fields for prediction
age = form.number_input(label="Age")
gender = form.radio("Gender", ["Male", "Female"])
bmi = form.number_input(label="BMI")
smoking = form.radio("Smoking", ["Yes", "No"])
genetic_risk = form.selectbox("Genetic Risk", ["Low", "Moderate", "High"])
physical_activity = form.number_input(label="Physical Activity (Hours/Week)")
alcohol_intake = form.radio("Alcohol Intake", ["Yes", "No"])
cancer_history = form.radio("Family Cancer History", ["Yes", "No"])

# Algorithm selection
Select = form.selectbox("Algorithm", ["KNN", "Naive Bayes", "Decision Tree"])
submit_button = form.form_submit_button(label='Submit')

# Convert categorical values to numeric
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
alcohol_intake = 1 if alcohol_intake == "Yes" else 0
cancer_history = 1 if cancer_history == "Yes" else 0

genetic_risk_mapping = {"Low": 0, "Moderate": 1, "High": 2}
genetic_risk = genetic_risk_mapping[genetic_risk]

# If the form is submitted
if submit_button:
    input_data = (int(age), int(gender), float(bmi), int(smoking), int(genetic_risk), 
                  int(physical_activity), int(alcohol_intake), int(cancer_history))
    
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # KNN Algorithm
    if Select == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        prediction = model.predict(input_data_as_numpy_array)
        accuracy = accuracy_score(Y_test, model.predict(X_test))

    # Naive Bayes Algorithm
    elif Select == "Naive Bayes":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X_train, Y_train)
        prediction = model.predict(input_data_as_numpy_array)
        accuracy = accuracy_score(Y_test, model.predict(X_test))

    # Decision Tree Algorithm
    elif Select == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        prediction = model.predict(input_data_as_numpy_array)
        accuracy = accuracy_score(Y_test, model.predict(X_test))

    # Display Prediction & Accuracy
    if prediction[0] == 1:
        st.subheader("Result: High Cancer Risk")
    else:
        st.subheader("Result: Low Cancer Risk")
    
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
