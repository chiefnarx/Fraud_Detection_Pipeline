from airflow.models import DagBag

def test_dag_integrity():
    dag_bag = DagBag(dag_folder="scripts/fdp_airflow/dags")
    assert len(dag_bag.import_errors) == 0, f"DAG import failures: {dag_bag.import_errors}"