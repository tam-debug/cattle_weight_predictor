from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from weight_model.k_fold_builder import load_folds_configuration

sns.set_theme()

def plot_weights_histogram(weights_file_path: Path):
    df = pd.read_csv(weights_file_path)

    plt.figure(figsize=(12, 6))
    sns.histplot(df["Weight"], binwidth=20)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Histogram of Weights')

    plt.show()

def plot_weights_bar(weights_file_path: Path):
    df = pd.read_csv(weights_file_path)
    bins = pd.cut(df['Weight'], bins=range(int(df['Weight'].min()), int(df['Weight'].max()) + 10, 10))
    weight_counts = bins.value_counts().sort_index()

    # Convert to a DataFrame for plotting
    weight_df = pd.DataFrame({'Weight Range': weight_counts.index.astype(str), 'Count': weight_counts.values})

    # Plot using a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Weight Range', y='Count', data=weight_df)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.title('Distribution of Weights')
    plt.tight_layout()

    plt.show()

def plot_weights_scatter(weights_file_path: Path):
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(weights_file_path)

    # Increase plot size
    plt.figure(figsize=(12, 6))

    # Create a scatter plot
    sns.scatterplot(x='Name', y='Weight', data=df, color="blue")

    # Add labels and title
    plt.xlabel('Cattle image')
    plt.ylabel('Weight')
    plt.title("Cattle weights dataset")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_histogram(data, title, colour=None):
    sns.histplot(data, binwidth=10, color=colour)

    # Add labels and legend
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(title)

    # Show plot
    plt.show()


def plot_fold_weights_histogram(weights_file_path: Path, k_folds_path: Path):

    df = pd.read_csv(weights_file_path)
    folds = load_folds_configuration(k_folds_path)
    for i, values in enumerate(folds, start=1):
        train_indexes, test_indexes = values
        plot_histogram(data=df["Weight"].iloc[train_indexes], title=f"Split {i} Train"),
        plot_histogram(data=df["Weight"].iloc[test_indexes], title=f"Split {i} Test", colour=sns.color_palette()[1])

def count_fold_instances(k_folds_path: Path):
    folds = load_folds_configuration(k_folds_path)
    data = []

    for i, values in enumerate(folds, start=1):
        train_indexes, test_indexes = values
        data.append([i, len(train_indexes), len(test_indexes)])

    df = pd.DataFrame(data, columns=["Split Number", "Train Instances", "Test Instances"])
    print(df)
