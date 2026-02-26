import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(data):

    folder = os.path.join(os.getcwd(), "src", "visualization")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "figures")

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.4)

    # Target count
    sns.countplot(data=data, x="target", hue="target", ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Distribution of Target")

    # Failure distribution
    sns.countplot(
        data=data, x="failure_type", hue="failure_type", ax=axes[0, 1], palette="magma"
    )
    axes[0, 1].set_title("Breakdown by Failure Type")

    # Correlation
    numeric_df = data.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title('Correlation')
    
    # Tool Wear vs Target (Bottom Right)

    sns.boxplot(
        data=data,
        x="target",
        y="tool_wear_min",
        hue="target",
        ax=axes[1, 1],
        palette="viridis",
        legend=False,
    )
    axes[1, 1].set_title("Tool Wear (min) vs. Failure")

    plt.savefig(file_path)
    plt.show()


if __name__ == "__main__":
    file_name = os.path.join(os.getcwd(), "data", "interim", "cleaned_df.csv")
    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
        plot_feature_importance(data)
    else:
        print(f"Error: Could not find file at {file_name}")
