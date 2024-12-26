import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import string
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text,
                    lowercase=True,
                    remove_punctuation=True,
                    remove_stopwords=True,
                    stemming=False,
                    lemmatization=True,
                    custom_stopwords=None,
                    remove_numbers=False,
                    apply_regex=None):
    """
    Performs generic text preprocessing.

    Args:
        text (str): The input text string.
        lowercase (bool, optional): Convert text to lowercase. Defaults to True.
        remove_punctuation (bool, optional): Remove punctuation marks. Defaults to True.
        remove_stopwords (bool, optional): Remove common English stop words. Defaults to True.
        stemming (bool, optional): Apply stemming using Porter's algorithm. Defaults to False.
        lemmatization (bool, optional): Apply lemmatization using WordNet. Defaults to True.
        custom_stopwords (list, optional): A list of additional stop words to remove. Defaults to None.
        remove_numbers (bool, optional): Remove numerical characters. Defaults to False.
        apply_regex (list of tuples, optional): A list of tuples where each tuple contains
                                               a regex pattern and its replacement. Defaults to None.

    Returns:
        str: The preprocessed text.
    """

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Apply custom regex replacements
    if apply_regex:
        for pattern, replacement in apply_regex:
            text = re.sub(pattern, replacement, text)

    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        tokens = [word for word in tokens if word not in stop_words]

    # Stemming or Lemmatization
    if stemming and lemmatization:
        raise ValueError("Cannot apply both stemming and lemmatization. Choose one.")

    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text



def preprocess_data(df: pd.DataFrame, imputation_strategies: dict = None) -> pd.DataFrame:
    """
    Preprocesses the idealized dataset, focusing on NLP tasks
    and potential use with LLMs and agents.

    Args:
        df: pandas DataFrame containing the raw data.
        imputation_strategies: A dictionary specifying the imputation strategy for each column
                               with missing values. Keys are column names, values are imputation
                               strategies ('mean', 'median', 'mode', or a constant value).
                               If None, defaults to median imputation for numerical and empty string for text.

    Returns:
        pandas DataFrame with preprocessed data.
    """

    df_processed = df.copy()

    # --- 1. Handle Missing Values with Customizable Imputation ---
    if imputation_strategies is None:
        imputation_strategies = {}  # Default strategies will be applied

    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            strategy = imputation_strategies.get(col)

            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
                df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                print(f"Missing values in '{col}' imputed with mean.")
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
                df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                print(f"Missing values in '{col}' imputed with median.")
            elif strategy == 'mode':
                # Before imputing with mode, ensure the column doesn't have mixed comparable types leading to NoneType comparison
                if pd.api.types.is_numeric_dtype(df_processed[col]) or df_processed[col].dtype == 'object':
                    try:
                        imputer = SimpleImputer(strategy='most_frequent')
                        df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                        print(f"Missing values in '{col}' imputed with mode.")
                    except TypeError as e:
                        print(f"Warning: Could not impute '{col}' with mode due to: {e}. Consider converting to a consistent type or using a different strategy.")
                else:
                    print(f"Warning: Cannot impute '{col}' with mode as it's not a recognized comparable type. Skipping mode imputation.")
            elif isinstance(strategy, (int, float, str)):  # Constant imputation
                df_processed[col] = df_processed[col].fillna(strategy)
                print(f"Missing values in '{col}' imputed with constant value: {strategy}.")
            elif strategy == 'drop_row':
                df_processed.dropna(subset=[col], inplace=True)
                print(f"Rows with missing values in '{col}' dropped.")
            elif strategy == 'drop_column':
                df_processed.drop(columns=[col], inplace=True)
                print(f"Column '{col}' with missing values dropped.")
            elif pd.api.types.is_numeric_dtype(df_processed[col]):
                # Default to median for numerical columns if no strategy is specified
                imputer = SimpleImputer(strategy='median')
                df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
                print(f"Warning: Missing values in numerical column '{col}' imputed with default median.")
            elif df_processed[col].dtype == 'object':
                # Default to empty string for object columns if no strategy is specified
                df_processed[col] = df_processed[col].fillna('')
                print(f"Warning: Missing values in object column '{col}' imputed with default empty string.")
            else:
                print(f"Warning: Missing values in '{col}'. No imputation strategy provided, leaving as is.")

    # --- 2. Text Preprocessing (Focus on NLP aspects) ---
    text_columns = [col for col in df_processed.columns if df_processed[col].dtype == 'object']
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()  # Or use WordNetLemmatizer for lemmatization

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        # tokens = [stemmer.stem(word) for word in tokens if word not in stop_words] # Using stemming here
        return " ".join(tokens)

    for col in text_columns:
        df_processed[col + '_processed'] = df_processed[col].apply(preprocess_text)

    # --- 3. Numerical Feature Scaling ---
    numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()

    # Remove any identifier columns or target variables you don't want to scale
    # Example: Assuming 'trade_id' is an identifier
    if 'trade_id' in numerical_features:
        numerical_features.remove('trade_id')

    # Filter out columns that might have been dropped due to imputation
    numerical_features = [col for col in numerical_features if col in df_processed.columns]

    if numerical_features:
        scaler = StandardScaler() # Or MinMaxScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    # --- 4. Categorical Feature Encoding ---
    categorical_features = df_processed.select_dtypes(include='object').columns.tolist()
    # Remove any text columns that were just processed
    categorical_features = [col for col in categorical_features if not col.endswith('_processed')]

    # Filter out columns that might have been dropped due to imputation
    categorical_features = [col for col in categorical_features if col in df_processed.columns]

    if categorical_features:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(df_processed[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(categorical_features, axis=1), df_encoded], axis=1)

    # --- 5. Handling Date Features
    date_columns = [col for col in df_processed.columns if 'date' in col.lower()] # Adjust as needed
    # for date_col in date_columns:
        # if pd.api.types.is_datetime64_any_dtype(df_processed[date_col]):
            # df_processed[date_col + '_year'] = df_processed[date_col].dt.year
            # df_processed[date_col + '_month'] = df_processed[date_col].dt.month
            # df_processed[date_col + '_day'] = df_processed[date_col].dt.day
            # df_processed[date_col + '_dayofweek'] = df_processed[date_col].dt.dayofweek
            # Consider cyclical encoding for month and day if relevant
            # df_processed.drop(date_col, axis=1, inplace=True)


    return df_processed

