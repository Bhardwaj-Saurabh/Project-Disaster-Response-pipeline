import json
import plotly
from wordcloud import WordCloud
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Count of Categories
    category_count = df.iloc[:,4:].sum()
    category_names = list(category_count.index)
    
    cat = df.iloc[:,4:]
    cat_mean = cat.mean().sort_values(ascending=False)[1:11]
    cat_name = list(cat_mean.index)
    
    # Extract text data from the 'message' column
    text_data = ' '.join(df['message'])
    # Generate a Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # create visuals
    graphs = [
        # GRAPH 1 - distribution by category
        {
            'data': [
                {
                    'x': category_names,
                    'y': category_count,
                    'type': 'bar',
                    'marker': {
                        'color': 'green',  # Change bar color to green
                    }
                }
            ],

            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -90,
                },
                'width': 1000,
                'height': 700,
                'margin': {
                    'pad': 10,
                    'b': 150,
                },
            }
        },
        # GRAPH 2 - distribution by genere
        {
            'data': [
                {
                    'x': genre_names,
                    'y': genre_counts,
                    'type': 'bar',
                    'marker': {
                        'color': 'blue',  # Change bar color to blue
                    }
                }
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'tickangle': -90,  # Rotate x-axis labels for better readability
                },
                'width': 1000,
                'height': 700,
                'margin': {
                    'pad': 10,
                    'b': 150,
                },
            }
        },
        
        # GRAPH 3 - Top 10 Message Categories
        {
            'data': [
                {
                    'x': cat_name,
                    'y': cat_mean,
                    'type': 'bar',
                    'marker': {
                        'color': 'purple',  # Change bar color to purple
                    }
                }
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': -90,
                },
                'width': 1000,
                'height': 700,
                'margin': {
                    'pad': 10,
                    'b': 150,
                },
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()