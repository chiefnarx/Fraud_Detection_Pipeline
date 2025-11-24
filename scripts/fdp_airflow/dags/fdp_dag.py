from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.append('/opt/airflow/scripts')

from scripts.pipeline.data_extraction import run_extraction
from scripts.pipeline.feature_engineering import run_engineering
from scripts.pipeline.model_training import run_training
from scripts.pipeline.batch_prediction import run_prediction

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 11, 12),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fraud_detection_pipeline',
    default_args=default_args,
    schedule='@daily',
    catchup=False,
    description='End-to-end fraud detection pipeline for batch scoring',
) as dag:

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=run_extraction,
    )

    engineer = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_engineering,
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=run_training,
    )

    predict = PythonOperator(
        task_id='batch_prediction',
        python_callable=run_prediction,
    )

    extract >> engineer >> train >> predict