import streamlit as st
import pickle

# Título de la app
st.title("Predicción de Ventas de Helados con Gradient Boosting")

# Cargar datos
# st.subheader("Cargando datos")



with open("../modelo_final.pkl", 'rb') as model_file:
    model = pickle.load(model_file)
with open("../scaler_modelo_final.pklscaler_modelo_final.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
temperatura_media = st.number_input("Introduce la temperatura media:")
humedad_relativa = st.number_input("Introduce la humedad relativa:")

input_scaled = scaler.transform([[temperatura_media, humedad_relativa]])

pred = model.predict(input_scaled)

st.write("Ventas previstas:", pred,"€")