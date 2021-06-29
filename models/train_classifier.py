import pandas as pd
import numpy as np
import nltk
import sys
import joblib
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath, database_table):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_table, engine)

    category_names = df.columns[4:]

    X = df['message'].values
    y = df[category_names].values

    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(n_jobs=1):
    forest = RandomForestClassifier()

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=forest)),
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=n_jobs)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(Y_test[0])):
        print("Classification report: '{}'".format(category_names[i]))
        print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    compressions = {
        'z': 'zlib',
        'gz': 'gzip',
        'bz2': 'bz2',
        'xz': 'xz',
        'lzma': 'lzma',
    }
    if model_filepath.split(".")[-1] in compressions:
        compression = (compressions[model_filepath.split(".")[-1]], 3)
        joblib.dump(model, model_filepath, compress=compression)
    else:
        joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) in [4, 5]:
        if len(sys.argv) == 4:
            database_filepath, database_table, model_filepath = sys.argv[1:]
            n_jobs = 1
        else:
            database_filepath, database_table, model_filepath, n_jobs = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}\n    TABLE:{}'.format(database_filepath, database_table))
        n_jobs = int(n_jobs)
        X, Y, category_names = load_data(database_filepath, database_table)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(n_jobs)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the table name in the second argument, '\
              'and the filepath of the pickle file to save the model to as the '\
              'second argument. \n\nExample: python train_classifier.py '\
              '../data/DisasterResponse.db messages classifier.pkl')


if __name__ == '__main__':
    main()