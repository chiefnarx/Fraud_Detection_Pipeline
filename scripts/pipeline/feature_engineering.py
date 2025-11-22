import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def run_engineering():
    try:
        # Load the raw data
        fdp_df = pd.read_csv('../dataset/raw/creditcard.csv')

        # Features = all columns except Class
        X = fdp_df.drop('Class', axis=1)

        # Target = only the Class column
        y = fdp_df['Class']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Undersample training data
        undersampler = RandomUnderSampler(random_state=42)
        X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train_scaled, y_train)

        # Save processed datasets
        pd.DataFrame(X_train_balanced).to_csv('../dataset/processed/X_train.csv', index=False)
        pd.DataFrame(X_test_scaled).to_csv('../dataset/processed/X_test.csv', index=False)
        pd.DataFrame(y_train_balanced).to_csv('../dataset/processed/y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('../dataset/processed/y_test.csv', index=False)

        print("Feature engineering completed successfully!")
        print(f"Balanced train shape: {X_train_balanced.shape}, Test shape: {X_test_scaled.shape}")

    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")