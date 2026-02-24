import pandas as pd


def check_class_imbalance(data: pd.DataFrame) -> bool:
    percentages = data.value_counts(normalize=True) * 100
    diff = percentages.max() - percentages.min()

    if diff > 20:
        return True

    return False


if __name__ == "__main__":
    file_name = r"data\raw\predictive_maintenance.csv"

    data = pd.read_csv(file_name)
    data = data["Target"]
    print(check_class_imbalance(data))
