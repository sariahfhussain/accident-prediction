import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Convert 'time' column to total minutes
    if 'time' in df.columns:
        df['time'] = df['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # Ensure the columns are named correctly
    df.rename(columns={
        'Day_of_week': 'day_of_week',
        'Area_accident': 'area_accident',
        'Type_of_vehicle': 'type_of_vehicle',
        'Lanes_or_Medians': 'lane_or_medians',
        'Road_surface_type': 'road_surface_type',
        'Road_surface_conditions': 'road_surface_conditions',
        'Light_conditions': 'light_conditions',
        'Weather_conditions': 'weather_conditions',
        'Sex_of_driver': 'sex_of_driver'
    }, inplace=True)

    # Label encoding for categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    correct_feature_names = ['time', 'day_of_week', 'area_accident', 'type_of_vehicle', 
                             'lane_or_medians', 'road_surface_type', 'road_surface_conditions', 
                             'light_conditions', 'weather_conditions', 'sex_of_driver']

    df = df[correct_feature_names]
    return df, label_encoders

def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
