import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

rating_matrix = pd.read_csv('Project4/I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv', sep=',')

def calculate_movie_popularity_scores():
    movie_metrics = {}
    for col in rating_matrix.columns:
        ratings = rating_matrix[col].dropna()
        if len(ratings) > 0:
            movie_metrics[col] = {
                'movie_id': col[1:],
                'rating_count': len(ratings),
                'average_rating': ratings.mean(),
                'popularity_score': len(ratings) * ratings.mean()
            }

    return pd.DataFrame.from_dict(movie_metrics, orient='index')

class MovieDatabase:
    def __init__(self):
        self._movies = {}
        self._load_movies()
    
    def _load_movies(self):
        """Load all movies into memory once"""
        try:
            with open('Project4/ml-1m/movies.dat', 'r', encoding='ISO-8859-1') as f:
                for line in f:
                    id_str, title_str, _ = line.split("::")
                    self._movies[id_str] = title_str.strip()
        except FileNotFoundError:
            print("Warning: movies.dat file not found")
    
    def get_movie_title(self, movie_id):
        """Get movie title from cache"""
        return self._movies.get(str(movie_id), "Unknown Title")
db = MovieDatabase()

def get_movie_title(movie_id):
    return db.get_movie_title(movie_id)

def get_movie_poster(movie_id):
    return f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"

def get_top_popular_movies(n=10):
    metrics_df = calculate_movie_popularity_scores()
    top_movies = metrics_df.nlargest(n, 'popularity_score')

    results = []
    for idx, row in top_movies.iterrows():
        results.append({
            'movie_id': 'm' + row['movie_id'],
            'title': get_movie_title(row['movie_id']),
            'rating_count': int(row['rating_count']),
            'average_rating': round(row['average_rating'], 2),
            'popularity_score': round(row['popularity_score'], 2),
            'poster_path': get_movie_poster(row['movie_id'])
        })

    return results

def normalize_rating_matrix(rating_matrix):
    return rating_matrix.subtract(rating_matrix.mean(axis=1), axis='rows')

"""#### Step 2: Build similarity matrix"""

def build_similarity_matrix(rating_matrix):

    normalized_matrix = normalize_rating_matrix(rating_matrix)

    normalized_matrix = normalized_matrix.T

    cardinality_df = (~normalized_matrix.isna()).astype('int')
    cardinality_matrix = cardinality_df @ cardinality_df.T

    normalized_matrix = normalized_matrix.fillna(0)

    nr = normalized_matrix @ normalized_matrix.T

    squared_matrix = ((normalized_matrix**2) @ (normalized_matrix!=0).T)
    squared_matrix = squared_matrix.apply(np.vectorize(np.sqrt))
    dr = squared_matrix * squared_matrix.T

    cosine_distance = nr/dr
    S = (1 + cosine_distance)/2

    np.fill_diagonal(S.values, np.nan)

    S[cardinality_matrix < 3] = np.nan

    return S

S = build_similarity_matrix(rating_matrix)

def filter_top_similarities(S, k=30):
    S_filtered = S.copy()

    for idx in S.index:
        row = S_filtered.loc[idx]
        non_na_values = row.dropna()

        if len(non_na_values) > k:
            kth_largest = non_na_values.nlargest(k).iloc[-1]
            mask = (row >= kth_largest) & row.notna()
            row[~mask] = np.nan

        S_filtered.loc[idx] = row

    return S_filtered

S_filtered = filter_top_similarities(S, k=100)

def get_popular_movie_ratings():
    movie_rating_stats = rating_matrix.groupby('MovieID').agg({
        'Rating': ['mean', 'count']
    }).droplevel(0, axis=1)

    mean_rating_count = movie_rating_stats['count'].mean()
    min_rating = movie_rating_stats['mean'].min()
    movie_rating_stats['weighted_rating'] = (movie_rating_stats['count'] * movie_rating_stats['mean'] +
                                           mean_rating_count * min_rating) / (movie_rating_stats['count'] + mean_rating_count)

    return movie_rating_stats['weighted_rating'].sort_values(ascending=False)

def get_popular_movie_ratings():
    movie_rating_stats = rating_matrix.agg(['mean', 'count'], axis=0)
    
    movie_rating_stats.columns = movie_rating_stats.columns.droplevel(0)
    
    mean_rating_count = movie_rating_stats.loc[:, 'count'].mean()
    min_rating = movie_rating_stats.loc[:, 'mean'].min()
    
    movie_rating_stats['weighted_rating'] = (
        movie_rating_stats['count'] * movie_rating_stats['mean'] +
        mean_rating_count * min_rating
    ) / (movie_rating_stats['count'] + mean_rating_count)
    
    return movie_rating_stats['weighted_rating'].sort_values(ascending=False)

def myIBCF(user_ratings, num_recommendations=10):

    working_similarities = S_filtered.copy()
    user_rating_vector = pd.Series(user_ratings)

    rated_movie_mask = ~user_rating_vector.isna()

    rating_predictions = {}
    for candidate_movie in working_similarities.index:
        if rated_movie_mask[candidate_movie]:
            continue

        movie_similarities = working_similarities.loc[candidate_movie][rated_movie_mask]
        existing_ratings = user_rating_vector[rated_movie_mask]

        if len(movie_similarities) == 0:
            continue

        predicted_rating = (movie_similarities * existing_ratings).sum() / movie_similarities.sum()
        rating_predictions[candidate_movie] = predicted_rating

    prediction_series = pd.Series(rating_predictions).sort_values(ascending=False)

    if len(prediction_series) < num_recommendations:
        print("Backfilling with popular movies...")
        popular_movie_ratings = get_popular_movie_ratings()
        unrated_popular_movies = popular_movie_ratings[~rated_movie_mask]
        needed_recommendations = num_recommendations - len(prediction_series)
        prediction_series = pd.concat([prediction_series, unrated_popular_movies[:needed_recommendations]])

    return prediction_series[:num_recommendations]