import sys
import json
import plotly
import joblib
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Heatmap
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    category_names = df.columns[4:]
    category_counts = np.sum(df[category_names].values, axis=0)
    category_counts_df = pd.DataFrame(list(zip(category_names, category_counts)), columns=["Category", "Counts"])
    category_counts_df = category_counts_df.sort_values(by="Counts", ascending=False)

    n_category_count = np.sum(df[category_names].values, axis=1)
    np.histogram(n_category_count)

    n_categories = len(category_names)
    cat_heat_map = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            if i != j:
                cat_heat_map[i, j] = np.sum((df[category_names[i]] == 1) & (df[category_names[j]] == 1))

    n_categories = len(category_names)
    cat_heat_map = np.zeros((n_categories, n_categories))
    categories_i = list(range(n_categories))
    for i in categories_i:
        for j in categories_i:
            if i != j:
                cat_heat_map[i, j] = np.sum((df[category_names[i]] == 1) & (df[category_names[j]] == 1))


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_counts_df['Category'].values[:20],
                    y=category_counts_df['Counts'].values[:20],
                )
            ],

            'layout': {
                'title': 'Top 20 Message Categories',
                'height': 450,
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True,
                    'tickangle': 40,
                },
            },
        },
        {
            'data': [
                Histogram(
                    x = n_category_count
                )
            ],

            'layout': {
                'title': 'Number of Categories per Message',
                'height': 450,
                'yaxis': {
                    'title': "Number of Messages",
                },
                'xaxis': {
                    'title': "Number of Categories",
                },
            },
        },
        {
            'data': [
                Heatmap(
                    z = cat_heat_map
                )
            ],

            'layout': {
                'title': 'Categories most likely to appear together',
                'height': 800,
                'yaxis': {
                    'tickvals': categories_i,
                    'ticktext': category_names,
                    'automargin': True,
                },
                'xaxis': {
                    'tickvals': categories_i,
                    'ticktext': category_names,
                    'tickangle': 30,
                    'automargin': True,
                },
            },
            'class': "col-lg-12",
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    classes = []
    for graph in graphs:
        if 'class' in graph:
            classes.append(graph['class'])
            del graph['class']
        else:
            classes.append("col-lg-6 col-sm-12")
    ids_classes = list(zip(ids, classes))
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids_classes=ids_classes, ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    if len(sys.argv) == 4:
        database_filepath, database_table, model_filepath = sys.argv[1:]
        # load data
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        global df
        df = pd.read_sql_table(database_table, engine)

        # load model
        global model
        model = joblib.load(model_filepath)

        app.run(host='0.0.0.0', port=3001, debug=True)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the table name in the second argument, '\
              'and the filepath of the pickle file where the model is stored '\
              'to as the second argument. \n\nExample: python train_classifier.py '\
              '../data/DisasterResponse.db messages classifier.pkl')


if __name__ == '__main__':
    main()