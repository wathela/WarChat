from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta


default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 2, 16), 
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "etl_embed_conflictsEvents_dag",
    default_args=default_args,
    schedule='@daily', # "*/7 * * * *", 
    catchup=False,) as dag:

    run_spark = BashOperator(
        task_id='etl_embed_spark_job',
        bash_command='spark-submit get_ingest_embed_spark_job.py',
    )

    run_spark  



