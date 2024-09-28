from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


dag = DAG(
    'data_prepare_dag',
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
)

prepare_task = BashOperator(
    task_id='prepare_data_task',
    bash_command='cd /home/amir/MlOps; python3 code/datasets/data_pipeline.py ',
    dag=dag,
)

prepare_task
