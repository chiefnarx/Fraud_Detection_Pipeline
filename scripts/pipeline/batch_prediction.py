import pandas as pd
import joblib

def run_prediction():
    try:
        # Load model and scaler
        model = joblib.load('../models/random_forest_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')

        # Load new batch data
        new_data = pd.read_csv('../dataset/raw/new_transactions.csv')

        # Preprocess
        new_data_scaled = scaler.transform(new_data)

        # Predict
        probas = model.predict_proba(new_data_scaled)[:, 1]
        threshold = 0.7
        predicted_labels = (probas >= threshold).astype(int)

        # Save predictions
        new_data['is_fraud'] = predicted_labels
        new_data.to_csv('../dataset/predictions/batch_predictions.csv', index=False)

        print("Batch prediction completed successfully!")
        print(f"Fraudulent transactions flagged: {predicted_labels.sum()}")

    except Exception as e:
        print(f"An error occurred during batch prediction: {e}")