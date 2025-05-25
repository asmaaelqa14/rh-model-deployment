import streamlit as st
import joblib
import pandas as pd



@st.cache_resource
def load_model():
    return joblib.load("final_pipeline_model.pkl")

model = load_model()



# Titre de l'application
st.title("🧠 Prédiction RH - Départ d'un employé")

# Charger le modèle
@st.cache_resource
def load_model():
    return joblib.load("full_pipeline_model.pkl")

model = load_model()

# Formulaire utilisateur
st.header("📋 Données de l'employé à prédire")

age = st.slider("Âge", 18, 60, 30)
total_working_years = st.slider("TotalWorkingYears", 0, 40, 10)
years_at_company = st.slider("YearsAtCompany", 0, 40, 5)
years_with_curr_manager = st.slider("YearsWithCurrManager", 0, 17, 3)
mean_presence_time = st.slider("meanPresenceTime", 5.0, 11.0, 7.5)

education_field = st.selectbox("EducationField", ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
business_travel = st.selectbox("BusinessTravel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
environment_satisfaction = st.selectbox("EnvironmentSatisfaction", [1.0, 2.0, 3.0, 4.0])
marital_status = st.selectbox("MaritalStatus", ['Married', 'Single', 'Divorced'])

# Créer un DataFrame avec les entrées utilisateur
input_data = pd.DataFrame([{
    'Age': age,
    'TotalWorkingYears': total_working_years,
    'YearsAtCompany': years_at_company,
    'YearsWithCurrManager': years_with_curr_manager,
    'meanPresenceTime': mean_presence_time,
    'EducationField': education_field,
    'BusinessTravel': business_travel,
    'EnvironmentSatisfaction': environment_satisfaction,
    'MaritalStatus': marital_status
}])

# Afficher les données saisies
st.subheader("📊 Données saisies :")
st.write(input_data)

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)
    st.success(f"✅ Résultat de la prédiction : {prediction[0]}")
