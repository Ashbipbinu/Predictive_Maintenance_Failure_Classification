import pandas as pd
import os
import yaml
import mlflow
import mlflow.sklearn


from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Predictive_Maintenance")



with mlflow.start_run(run_name="Parent Optimization Run") as parent_run:
    
    print(f"parent run id: {parent_run.info.run_id}")

