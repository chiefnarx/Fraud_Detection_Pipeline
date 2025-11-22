import pandas as pd

def run_extraction():
    try:
        fdp_df = pd.read_csv("../dataset/raw/creditcard.csv")
        print("Data loaded successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")