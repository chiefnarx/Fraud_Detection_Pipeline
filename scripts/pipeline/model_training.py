import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def run_training():
    try:
        # Load processed data
        X_train = pd.read_csv('../dataset/processed/X_train.csv')
        X_test = pd.read_csv('../dataset/processed/X_test.csv')
        y_train = pd.read_csv('../dataset/processed/y_train.csv').values.ravel()
        y_test = pd.read_csv('../dataset/processed/y_test.csv').values.ravel()

        # Logistic Regression
        print("Training Logistic Regression...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("Logistic Regression Results:")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # Threshold optimization
        threshold = 0.7
        y_pred_opt = (y_pred_proba >= threshold).astype(int)
        print(f"\nOptimized Threshold ({threshold}):")
        print(classification_report(y_test, y_pred_opt))
        print(confusion_matrix(y_test, y_pred_opt))

        # Save model
        joblib.dump(model, '../models/logistic_model.pkl')

        # Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_pred_rf_proba = rf_model.predict_proba(X_test)[:, 1]

        print("Random Forest Results:")
        print(classification_report(y_test, y_pred_rf))
        print(confusion_matrix(y_test, y_pred_rf))

        # Threshold optimization
        y_pred_rf_opt = (y_pred_rf_proba >= threshold).astype(int)
        print(f"\nOptimized Threshold ({threshold}):")
        print(classification_report(y_test, y_pred_rf_opt))
        print(confusion_matrix(y_test, y_pred_rf_opt))

        # Save model
        joblib.dump(rf_model, '../models/random_forest_model.pkl')

        print("\nModel training completed successfully!")

    except Exception as e:
        print(f"An error occurred during model training: {e}")