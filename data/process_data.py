'''
Create ETL pipeline to save final dataset to SQLalchemy.

Author: Saurabh Bhardwaj
Date: 25 Sep 2023
'''

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    Args:
        messages_filepath (str): The file path to the messages dataset CSV file.
        categories_filepath (str): The file path to the categories dataset CSV file.
    Returns:
        dataset (pandas.DataFrame): Merged dataset containing messages and categories.
    """
    # Load the messages dataset 
    messages = pd.read_csv(messages_filepath)
    # Load the categories dataset
    categories = pd.read_csv(categories_filepath)
    # Merge the datasets on the 'id' column using an inner join
    dataset = messages.merge(categories, on='id', how='inner')
    # Return the merged dataset
    return dataset


def clean_data(dataframe):
    """
    Clean and preprocess the input dataframe.
    Args:
        dataframe (pandas.DataFrame): The input dataframe to be cleaned.
    Returns:
        cleaned_dataframe (pandas.DataFrame): The cleaned and preprocessed dataframe.
    """
    # Create a dataframe of the 36 individual category columns
    categories = dataframe['categories'].str.split(';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # Extract a list of new column names for categories
    category_colnames = [i.split('-')[0] for i in row]
    # Rename the columns of `categories`
    categories.columns = category_colnames
    # Iterate through each column in `categories`
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1])
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Drop the original 'categories' column from the input dataframe
    dataframe.drop('categories', axis=1, inplace=True)
    # Concatenate the original dataframe with the new `categories` dataframe
    dataframe = pd.concat([dataframe, categories], axis=1)
    # Remove duplicate rows
    dataframe = dataframe.drop_duplicates()
    # Return dataframe
    return dataframe


def save_data(dataframe, database_filename):
    """
    Save a DataFrame to a SQLite database.
    Args:
        dataframe (pandas.DataFrame): The DataFrame to be saved to the database.
        database_filename (str): The file path for the SQLite database.
    Returns:
        None
    """
    # Create an SQLAlchemy engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    # Save the DataFrame
    dataframe.to_sql('InsertTableName', engine, index=False)


def main():
    '''
    Run Data Pipeline
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        dataframe = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        dataframe = clean_data(dataframe)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(dataframe, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()