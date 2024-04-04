# DataPrep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def preprocess_data():
    # Load the dataset with explicit column names
    train_data = pd.read_csv('train.csv', names=['statement', 'label'])
    test_data = pd.read_csv('test.csv', names=['statement', 'label'])
    valid_data = pd.read_csv('valid.csv', names=['statement', 'label'])

    # Combine the datasets
    data = pd.concat([train_data, test_data, valid_data])

    # Print the columns of the combined dataset
    print(data.columns)

    # Check for NaN values in the 'label' column
    print(data['label'].isnull().sum())

    # Print the unique values in the 'label' column before mapping
    print(data['label'].unique())

    # Drop rows with NaN labels
    data = data.dropna(subset=['label'])

    # Preprocess the data
    data['label'] = data['label'].map({'true': 'True', 'mostly-true': 'True', 'half-true': 'True',
                                       'barely-true': 'False', 'false': 'False', 'pants-fire': 'False'})

    # Print the unique labels and their value counts after mapping
    print(data['label'].unique())
    print(data['label'].value_counts())

    # Check if the 'label' column is empty after mapping
    if data['label'].empty:
        raise ValueError(
            "The 'label' column is empty after label mapping. Please check the format of the labels in your CSV files.")

    # Encode the labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Split the data into train, test, and validation sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Extract the statements and labels
    X_train = train_data['statement']
    y_train = train_data['label']
    X_test = test_data['statement']
    y_test = test_data['label']
    X_valid = valid_data['statement']
    y_valid = valid_data['label']

    # Perform feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid)

    return X_train_tfidf, y_train, X_test_tfidf, y_test, X_valid_tfidf, y_valid, tfidf_vectorizer