"""


Akshay 


"""

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the anime dataset
anime = pd.read_csv(r"C:\Users\MyVampire\Desktop\AML project\Movie Recommendation\anime.csv", encoding='utf8')

# Create a TfidfVectorizer and fit it on the anime genre
tfidf = TfidfVectorizer(stop_words="english")
anime["genre"] = anime["genre"].fillna(" ")
tfidf_matrix = tfidf.fit_transform(anime.genre)

# Compute the cosine similarity matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping of anime name to index number
anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicates()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = get_recommendations(movie_name, topN=10)
        return render_template('index.html', movie_name=movie_name, recommendations=recommendations)
    return render_template('index.html')


def get_recommendations(name, topN):
    anime_id = anime_index[name]
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    cosine_scores_N = cosine_scores[0:topN+1]
    anime_idx = [i[0] for i in cosine_scores_N]
    anime_scores = [i[1] for i in cosine_scores_N]
    anime_similar_show = pd.DataFrame(columns=["name", "Score"])
    anime_similar_show["name"] = anime.loc[anime_idx, "name"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.reset_index(inplace=True)
    return anime_similar_show.values.tolist()


if __name__ == '__main__':
    app.run()
