'''
A package to train a model to classify disaster messages.

Author: Saurabh Bhardwaj
date: 25 Sep 2023
'''

# import libraries
import sys
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from a SQLite database.
    Args:
        database_filepath (str): The file path to the SQLite database.
    Returns:
        X (pandas.Series): A pandas Series containing the 'message' column.
        Y (pandas.DataFrame): A pandas DataFrame containing the target variables.
        Category: Data Column Labels , a list of the column names
    """
    # Create an SQLAlchemy engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    # Read the data from the specified table in the database
    df = pd.read_sql_table('InsertTableName', con=engine)
    # Separate the features (X) and target variables (Y)
    X = df['message']  # features
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)  # targets
    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenize and preprocess text data.
    Args:
        text (str): The input text to be tokenized and preprocessed.
    Returns:
        list of str: A list of preprocessed tokens.
    """
    # Text Cleaning
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenization
    words = word_tokenize(text)
    # Stopword Removal
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    # Return the preprocessed tokens
    return lemmed 


def build_model():
    """
    Build a text classification model pipeline.
    Args:
        None
    Returns:
        model: A scikit-learn estimator representing the best model from hyperparameter tuning.
    """
    # Define a pipeline with text processing and classification components
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Text vectorization
        ('tfidf', TfidfTransformer()),  # TF-IDF transformation
        ('clf', MultiOutputClassifier(MultinomialNB()))  # MultiOutputClassifier with Naive Bayes
    ])
    # Define a hyperparameter grid for grid search
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],  # Adjust the range of n-grams to consider
        'tfidf__use_idf': [True, False],  # Whether to use inverse document frequency (TF-IDF)
        'clf__estimator__alpha': [0.1, 0.5, 1.0]  # Alpha parameter for Multinomial Naive Bayes
    }
    # Perform grid search for hyperparameter tuning
    model = GridSearchCV(pipeline, parameters, cv=3)  # 3-fold cross-validation
    # Return the best estimator (model with optimized hyperparameters)
    return model
    
   
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a text classification model and calculate F1 scores for each category.
    Args:
        model (estimator): A trained scikit-learn estimator.
        X_test (pandas.Series): Test features.
        Y_test (pandas.DataFrame): True labels for test data.
        category_names (list): List of category names.
    Returns:
        f1_score_result (dict): A dictionary containing F1 scores for each category.
    """
    # Make predictions on the test data using the model
    Y_pred = model.predict(X_test)
    # Initialize a dictionary to store F1 scores for each category
    f1_score_result = {}
    # Iterate through each category
    for i, category in enumerate(category_names):
        # Calculate the weighted F1 score for the category
        f1_score_result[category] = f1_score(Y_test[category], Y_pred[:, i], average='weighted')
    # Print the F1 scores and return the results
    f1_scores_series = pd.Series(f1_score_result)
    print(f1_scores_series)
    return f1_score_result

def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file.
    Args:
        model (estimator): A trained scikit-learn estimator.
        model_filepath (str): The file path where the model will be saved.
    Returns:
        None
    """
    try:
        # Use pickle to save the trained model
        with open(model_filepath, 'wb') as file:  
            pickle.dump(model, file)
    except Exception as err:
        # Handle any exceptions
        print(f"An error occurred while saving the model: {str(e)}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()