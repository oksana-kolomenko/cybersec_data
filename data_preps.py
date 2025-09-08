import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from csv_parser import patient_info, number_to_word
from values import DatasetName, Textstyle


# Load features
def load_features(file_path, delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter)
    print(f"features: {data}")
    return data


def load_labels(file_path, delimiter=','):
    # Load the labels
    data = pd.read_csv(file_path, delimiter=delimiter)
    labels = data.values.ravel()  # Flatten in case it's a single column DataFrame
    return labels


# Load features as text summaries (create if doesn't exist)
def load_summaries(file_name):
    if not os.path.exists(file_name):
        print("File not found")
        return []
    with open(file_name, "r") as file:
        summaries_list = [line.strip() for line in file.readlines()]
    return summaries_list


def write_summary(file_name, summaries):
    with open(file_name, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(summary + "\n")
    print(f"All patient summaries written")


def create_summary(dataset_name, table_file, summaries_file, text_style):
    if dataset_name == DatasetName.POSTTRAUMA.value:
        all_patient_summaries = create_patient_summaries(table_file, text_style)
        write_summary(file_name=summaries_file, summaries=all_patient_summaries)

    elif dataset_name == DatasetName.CYBERSECURITY.value:
        all_cybersecurity_summaries = create_general_summaries(table_file, text_style)
        write_summary(file_name=summaries_file, summaries=all_cybersecurity_summaries)

    elif dataset_name == DatasetName.LUNG_DISEASE.value:
        all_ld_summaries = create_general_summaries(table_file, text_style)
        write_summary(file_name=summaries_file, summaries=all_ld_summaries)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def create_patient_summaries(csv_path, text_style=Textstyle.ONE.value):

    df_tab_data = pd.read_csv(csv_path)
    summaries = []

    if text_style == Textstyle.ONE.value or text_style == Textstyle.TWO.value:
        for patient_number, (_, row) in enumerate(df_tab_data.iterrows(), 1):
            patient_info_n_values = row.to_dict()
            summary = ""
            if text_style == Textstyle.ONE.value:
                summary = (f"The following is the data for patient number {patient_number}. " +
                           patient_info(patient_info_n_values, csv_path))
            elif text_style == Textstyle.TWO.value:
                summary = (f"We want to predict health risks. The following is the data for patient number {patient_number}. " +
                           patient_info(patient_info_n_values, csv_path))
            summaries.append(summary)

    elif text_style == Textstyle.THREE.value:
        for _, row in df_tab_data.iterrows():
            # Format each row as key:value ; key:value
            summary = " ; ".join(f"{col}:{row[col]}" for col in df_tab_data.columns)
            summaries.append(summary)

    else:
        raise ValueError(f"Unknown text style: {text_style}")

    return summaries


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


def create_general_summaries_(tab_data, categorial_values=None, column_name_map=None, output_file=None):
    df_tab_data = pd.read_csv(tab_data)
    summaries = []

    # Fallbacks bei None
    categorial_values = categorial_values or {}
    column_name_map = column_name_map or {}

    # Erkennung numerischer Spalten
    numeric_cols = df_tab_data.select_dtypes(include=['float64', 'int64']).columns
    stats = {
        col: {
            "mean": df_tab_data[col].mean(),
            "std": df_tab_data[col].std()
        }
        for col in numeric_cols
    }

    for line_number, (_, row) in enumerate(df_tab_data.iterrows(), 1):
        summary = f"The following is the data for patient number {line_number}. "
        details = []

        for col in df_tab_data.columns:
            value = row[col]

            # Nur echte NaNs oder leere Werte überspringen, aber "None" als Text beibehalten
            if pd.isna(value):
                continue
            if isinstance(value, str) and value.strip() == '' and value.strip().lower() != 'none':
                continue

            # Freundlicher Spaltenname
            new_col = column_name_map.get(col, col.replace("_", " ").capitalize())

            if col in categorial_values:
                try:
                    value_str = categorial_values[col].get(int(value), str(value))
                except (ValueError, TypeError):
                    value_str = str(value)
                details.append(f"{new_col} is {value_str}")
            elif col in numeric_cols:
                classified = number_to_word(value, stats[col]['mean'], stats[col]['std'])
                details.append(f"{new_col} is {classified}")
            else:
                details.append(f"{new_col} is {value}")

        summary += "; ".join(details) + "."
        summaries.append(summary)

    if output_file:
        with open(output_file, "w") as f:
            for summary in summaries:
                f.write(summary + "\n")

    return summaries
