import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import preprocess_data

# Load the trained model and feature names
model, feature_names = joblib.load('models/accident_model.pkl')

# Define the main function
def main():
    st.title("Accident Prediction App")

    # Input fields
    time_hour = st.selectbox("Hour", [f"{i:02d}" for i in range(24)])
    time_minute = st.selectbox("Minute", [f"{i:02d}" for i in range(60)])
    day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    area_accident = st.selectbox("Area of Accident", [
        'Office areas', 'Recreational areas', 'Residential areas', 'Industrial areas', 
        'Other', 'Church areas', 'Market areas', 'Rural village areas', 'Outside rural areas', 
        'Hospital areas', 'School areas', 'Unknown', 'Rural village areasOffice areas'
    ])
    type_of_vehicle = st.selectbox("Type of Vehicle", [
        'Lorry (41–100Q)', 'Public (12 seats)', 'Ridden horse', 'Lorry (11–40Q)', 'Turbo', 'Taxi', 
        'Bicycle', 'Automobile', 'Other', 'Pick up up to 10Q', 'Public (13–45 seats)', 'Special vehicle', 
        'Stationwagen', 'Long lorry', 'Bajaj', 'Public (> 45 seats)', 'Motorcycle'
    ])
    lane_or_medians = st.selectbox("Lane or Medians", [
        'Undivided Two way', 'other', 'Double carriageway (median)', 'One way', 
        'Two-way (divided with solid lines road marking)', 'Two-way (divided with broken lines road marking)', 
        'Unknown'
    ])
    road_surface_type = st.selectbox("Road Surface Type", [
        'Asphalt roads', 'Earth roads', 'Gravel roads', 'Other', 'Asphalt roads with some distress'
    ])
    road_surface_conditions = st.selectbox("Road Surface Conditions", [
        'No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape'
    ])
    light_conditions = st.selectbox("Light Conditions", [
        'Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'
    ])
    weather_conditions = st.selectbox("Weather Conditions", [
        'Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist'
    ])
    sex_of_driver = st.selectbox("Sex of Driver", ["Male", "Female", "Other"])

    # Combine time input
    time = f"{time_hour}:{time_minute}"

    # Convert input into DataFrame
    input_data = pd.DataFrame({
        'time': [time],
        'day_of_week': [day_of_week],
        'area_accident': [area_accident],
        'type_of_vehicle': [type_of_vehicle],
        'lane_or_medians': [lane_or_medians],
        'road_surface_type': [road_surface_type],
        'road_surface_conditions': [road_surface_conditions],
        'light_conditions': [light_conditions],
        'weather_conditions': [weather_conditions],
        'sex_of_driver': [sex_of_driver]
    })

    # Preprocess the input data
    input_data, _ = preprocess_data(input_data)

    # Ensure the input data columns match the feature names
    input_data = input_data[feature_names]

    # Make prediction
    if st.button("Predict"):
        prediction_proba = model.predict_proba(input_data)
        accident_chance = prediction_proba[0][1] * 100  # Assuming the second column is the probability of accident

        st.write(f"The predicted accident chance is: {accident_chance:.2f}%")

if __name__ == "__main__":
    main()
