import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#from csv_parser import patient_info, number_to_word
from values import DatasetName, Textstyle


# Load features
def load_features(file_path, delimiter=',', n_samples=None):
    data = pd.read_csv(file_path, delimiter=delimiter)
    if n_samples:
        data = data.head(n_samples)  # Take only the first n_samples rows
    print(f"features: {data}")
    return data


def load_labels(file_path, delimiter=','):
    # Load the labels
    data = pd.read_csv(file_path, delimiter=delimiter)
    y = data.values.ravel()  # Flatten in case it's a single column DataFrame

    y = pd.Series(y)
    if not np.issubdtype(y.dtype, np.number):
        print(f"Label encoding: {y.unique()}")
        le = LabelEncoder()
        y = le.fit_transform(y)  # this returns a NumPy array
    else:
        y = y.to_numpy()
    return y


def load_summaries(file_name, n_samples=None):
    if not os.path.exists(file_name):
        print("File not found")
        return []
    with open(file_name, "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    if n_samples:
        summaries_list = summaries_list[:n_samples]
    return summaries_list


def write_summary(file_name, summaries):
    with open(file_name, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(summary + "\n")
    print(f"All patient summaries written")


def create_general_summaries(tab_data): #, text_style):
    df_tab_data = pd.read_csv(tab_data)
    summaries = []

    for line_number, (_, row) in enumerate(df_tab_data.iterrows(), 1):
        summary = f"The following is the data for sample number {line_number}. "
        # summary = f"We want to predict whether patients will recover from their lung disease. The following is the data for patient number {line_number}: "

        details = []
        for col in df_tab_data.columns:
            value = row[col]
            if pd.isna(value) or value == '':
                continue  # Skip missing or empty values
            details.append(f"{col} is {value}")

        summary += "; ".join(details) + "."
        summaries.append(summary)

    return summaries
