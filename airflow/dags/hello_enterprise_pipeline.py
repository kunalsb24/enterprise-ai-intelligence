from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def start_pipeline():
    print("Enterprise AI pipeline started.")


def validate_data():
    print("Data validation completed successfully.")


def finish_pipeline():
    print("Enterprise AI pipeline finished.")


with DAG(
    dag_id="hello_enterprise_pipeline",
    description="First Airflow DAG for the Enterprise AI project",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["enterprise-ai", "learning"],
) as dag:

    start_task = PythonOperator(
        task_id="start_pipeline",
        python_callable=start_pipeline,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    finish_task = PythonOperator(
        task_id="finish_pipeline",
        python_callable=finish_pipeline,
    )

    start_task >> validate_task >> finish_task