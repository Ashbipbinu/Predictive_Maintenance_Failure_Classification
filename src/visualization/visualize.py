import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import dagshub
import mlflow

from src.utensil.load_config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def clean_and_convert_numeric(df, columns):
    """
    Cleans string-based numeric columns and converts them to float.
    Handles units (like 'K' or 'min') and extra whitespace.
    """
    for col in columns:
        if col in df.columns:

            # If the column is already numeric, this does nothing.
            # If it's a string, it strips whitespace
            # and removes common non-numeric chars.

            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(r"[^0-9.-]", "", regex=True)

            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def visualize_target_analysis(df, target_col="target", save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Generating target analysis plots")

    # 1. Full Correlation Matrix
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")

        corr_path = os.path.join(save_dir, "full_correlation_matrix.png")
        plt.savefig(corr_path)
        mlflow.log_artifact(corr_path)
        plt.close()

    # 2. Feature Distribution by Target (Boxplots)
    features_to_plot = ["air_temperature_k", "torque_nm", "rotational_speed_rpm"]
    for col in features_to_plot:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 5))
            sns.boxplot(
                data=df,
                x=target_col,
                y=col,
                hue=target_col,
                palette="Set2",
                legend=False,
            )
            plt.title(f"Distribution of {col} by {target_col}")

            box_path = os.path.join(save_dir, f"boxplot_{col}.png")
            plt.savefig(box_path)
            mlflow.log_artifact(box_path)
            plt.close()

    # 3. Countplot of target
    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df, x=target_col, hue=target_col, palette="viridis", legend=False
    )
    plt.title(f"Count Plot of {target_col}")
    plt.tight_layout()

    count_path = os.path.join(save_dir, "target_count_plot.png")
    plt.savefig(count_path)
    mlflow.log_artifact(count_path)
    plt.close()


def plot_features_vs_target(
    df,
    target_col="target",
    x_features=None,
    y_feature=None,
    hue=None,
    save_dir="reports/figures",
):
    os.makedirs(save_dir, exist_ok=True)
    if x_features is None or y_feature is None:
        raise ValueError("x_features and y_feature must be provided")

    hue = hue or target_col

    for x_col in x_features:
        if x_col not in df.columns or y_feature not in df.columns:
            logger.warning(f"Skipping {x_col} or {y_feature} - not found")
            continue

        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x=x_col, y=y_feature, hue=hue, fill=True, alpha=0.5)
        plt.title(f"Density: {y_feature} vs {x_col}")
        plt.tight_layout()

        file_name = f"density_{x_col}_vs_{y_feature}.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path)
        mlflow.log_artifact(file_path)
        plt.close()


if __name__ == "__main__":
    config = load_config("config.yaml")
    data_path = config["data"]["train_data"]

    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        target_col = "target"

        # --- REFINED DATA CLEANING ---
        cols_to_fix = [
            "air_temperature_k",
            "torque_nm",
            "rotational_speed_rpm",
            "tool_wear_min",
            "temp_diff_k",
            "target",
        ]
        data = clean_and_convert_numeric(data, cols_to_fix)
        data = data.dropna(subset=[target_col])
        # -----------------------------

        try:
            dagshub.init(
                repo_owner=os.getenv("DAGSHUB_USER_NAME"),
                repo_name=os.getenv("DAGSHUB_REPO_NAME"),
                mlflow=True,
            )

            with mlflow.start_run():
                visualize_target_analysis(data, target_col=target_col)

                plot_features_vs_target(
                    data,
                    x_features=[
                        "air_temperature_k",
                        "temp_diff_k",
                        "rotational_speed_rpm",
                    ],
                    y_feature="tool_wear_min",
                )

                if "failure_type" in data.columns:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(
                        data=data,
                        x="failure_type",
                        hue="failure_type",
                        palette="magma",
                        order=data["failure_type"].value_counts().index,
                        legend=False,
                    )
                    plt.title("Count of Each Failure Type")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    fail_path = os.path.join(
                        "reports/figures", "failure_type_counts.png"
                    )
                    plt.savefig(fail_path)
                    mlflow.log_artifact(fail_path)
                    plt.close()

        except Exception as e:
            logger.error(f"Execution failed: {e}")
    else:
        logger.error(f"Could not find file at {data_path}")
