import pandas as pd
from xgboost import XGBClassifier
import joblib
from data_preprocessing import preprocess_data, split_data

def train_model(data_path, target_column, model_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Preprocess the data
    df, label_encoders = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Train the model
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # Save the model and feature names
    joblib.dump((model, X_train.columns.tolist()), model_path)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_model('data/accident_data.csv', 'area_accident', 'models/accident_model.pkl')
