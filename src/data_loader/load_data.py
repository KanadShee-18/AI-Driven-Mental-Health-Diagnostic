"""
Data Loading Module
This module contains functions to load and display mental health data
"""

import pandas as pd
import os

# Load the data from specific file path:
def load_data(file_path):
    """
    First we've to load the data and return it as a pandas dataframe for further processing
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found in: {file_path}... Please provide valid data in the proper path!")
    
    # Converting .csv file data into a table like structure
    df = pd.read_csv(file_path)

    return df;


# Showing the data:

def show_data_info(df):
    """
    Printing a beautiful heading
    """
    print("=" * 60)
    print(" " * 15 + "DATASET INFORMATION" + " " * 15)
    print("=" * 60)

    # Printing the basic info about the dataset
    print(f"\n📊 Dataset Size:")
    print(f"    Total Samples (rows): {len(df)}")
    print(f"    Total Featured (columns): {len(df.columns)}")

    # Checking data type for each column
    print(f"\n📝 Columns:")
    for col in df.columns:
        data_type = df[col].dtype
        print(f"    - {col}: {data_type}")

    # Show first few rows:
    print(f"\n🫣 First 5 rows: ")
    print("-" * 60)
    print(df.head()) # it will show first 5 rows by default

    # Showing the last 5 rows
    print(f"\n🙄 Last 5 rows: ")
    print("-" * 60)
    print(df.tail());

    # Checking how many times each label appears:
    print(f"\n📈 Label Distribution: ")
    print("-" * 60)
    label_counts = df['label'].value_counts()

    # Printing each level and its count
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    {label:15s}: {count:3d} samples ({percentage:5.1f}%)")

    # Also checking for missing values:
    print(f"\n⚠️ Missing Value Check: ")
    print("-" * 60)
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print(f"    ✅ No missing values - Dataset is clean!")
    else:
        print(missing)

    # Checking some sample text length:
    print(f"\n🗨️ Text Length Statistics:")
    print("-" * 60)
    # Calculate length of each text
    df['text_length'] = df['text'].str.len()
    print(f"    Minimum text length: {df['text_length'].min()} characters")
    print(f"    Maximum text length: {df['text_length'].max()} characters")
    print(f"    Average text length: {df['text_length'].mean():.1f} characters")

    # printing the footer
    print("\n" + "=" * 60)
