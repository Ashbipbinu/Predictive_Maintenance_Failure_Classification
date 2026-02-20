import pandas as pd
import os

from src.utensil.check_class_imabalance import check_class_imbalance
from src.utensil.handle_imbalance import handle_imbalance
from src.utensil.save_file import save_file


def data_load_preprocessing(file_name: str):

    if not os.path.exists:
        print(f"Failed: file does not found at location {file_name} ")
        return None

    df = pd.read_csv(file_name)

    if not df.empty:
        print(f"Data fetched successfully, Shape: {df.shape}")

        # Dropping productid and UID
        columns = ["UDI", "Product ID"]
        df.drop(columns=columns, inplace=True)
        print("Successfully removed columns", columns)

        # Converting the Type [ the quality class of the machinery being monitored ]
        if "Type" in df.columns:
            df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

        # Check for any missing values
        if not df.isna().values.any():
            print("Not found any missing values")
        else:
            # Handling the missing columns of numerical values - replace missing ones with mean
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Fill categorical values with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            df[categorical_cols] = df[categorical_cols].fillna(
                df[categorical_cols].mode().iloc[0]
            )

        # Renaming the column names
        columns = [
            "Air temperature [K]",
            "Process temperature [K]Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Failure Type",
        ]
        # Removing [] brackets and return only charaters
        cleaned_columns = [col.replace("[", "").replace("]", "") for col in df.columns]
        cleaned_columns = [col.replace(" ", "_").lower() for col in cleaned_columns]
        df.columns = cleaned_columns

        # Checking class imbalance in the target- there are 2 targets in the particular data set - Target and Failure type
        is_Target_imbalanced = check_class_imbalance(df["target"])
        is_Failure_Type_imbalanced = check_class_imbalance(df["failure_type"])

        if is_Target_imbalanced or is_Failure_Type_imbalanced:
            balanced_df = handle_imbalance(df)
        
        print(balanced_df.dtypes)

    return balanced_df


directory = os.getcwd()
file_name = os.path.join(directory, r"data\raw\predictive_maintenance.csv")
df_cleaned = data_load_preprocessing(file_name)
save_file(directory, df_cleaned)
