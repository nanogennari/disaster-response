import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def split_column(df, column, marker=";", value_marker="-", value_type=int):
    """Expands columns in the format: "categrory - value; ...; category - value"
        on to multiple columns.

    Args:
        df (DataFrame): DataFrame containing the column.
        column (str): Column name.

    Returns:
        DataFrame: DataFrame with the column expanded.
    """
    # Split columns in to multiple columns
    df_split = df[column].str.split(marker, expand=True)

    # Extract and apply column names
    row = df_split.iloc[0]
    category_colnames = row.str.split(value_marker, expand=True)[0]
    df_split.columns = category_colnames

    # Extract values
    df_split = df_split.apply(
        lambda column: column.apply(
            lambda x: x.split(value_marker)[-1]
            ).astype(value_type), axis=1
        )

    # Merge new columns on to original DataFrame and drop original column
    df = df.merge(df_split, left_index=True, right_index=True)
    df = df.drop(columns=column)
    return df


def load_data(messages_filepath, categories_filepath):
    """Loads and merge messages and categories from csv files.

    Args:
        messages_filepath (str): Messages csv file path.
        categories_filepath (str): Categories csv file path.

    Returns:
        pd.DataFrame: DataFrame with the loaded data.
    """
    # Load messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge dataframes
    return messages.merge(categories, on="id")


def clean_data(df):
    """Expands categories, removes rows with non binary categories and duplicates.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned.

    Returns:
         pd.DataFrame: DataFrame with the cleaned data.
    """
    # Expand categories on to multiple columns
    df = split_column(df, "categories")

    # Remove non binary categories values, we don't know the meaning of such values
    category_names = df.columns[4:]
    df = df[np.any((df[category_names] == 0) | (df[category_names] == 1), axis=1)]

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename, database_table):
    """Saves data in a SQLite database.

    Args:
        df (pd.DataFrame): DataFrame to be stored.
        database_filename (str): Database file path.
        database_table (str): Database file to store data.
    """
    # Store on to a SQLite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_table, engine, index=False)


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, \
            database_filepath, database_table = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, database_table)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument and the table name as the forth argument. '\
              '\n\nExample: python process_data.py disaster_messages.csv '\
              'disaster_categories.csv DisasterResponse.db messages')


if __name__ == '__main__':
    main()