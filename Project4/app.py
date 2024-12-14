import streamlit as st
import pandas as pd
from myIBCF import get_top_popular_movies, myIBCF, S, get_movie_title, get_movie_poster

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Rate some movies and get personalized recommendations!")

@st.cache_data
def load_sample_movies():
    return get_top_popular_movies()

movies_df = load_sample_movies()

st.subheader("Rate These Movies")
st.write("Please rate the movies you've watched (1-5 stars)")

user_ratings = {}
cols = st.columns(3)
for idx, movie in enumerate(movies_df):
    with cols[idx % 3]:
        st.write(f"### {movie['title']}")
        st.image(movie['poster_path'], caption=movie['title'])
        rating = st.select_slider(
            f"Rate {movie['title']}",
            options=['Not seen', '1', '2', '3', '4', '5'],
            key=f"movie_{movie['movie_id']}"
        )
        if rating != 'Not seen':
            user_ratings[movie['movie_id']] = int(rating)

def parse_user_ratings(rating_dict):
    user_ratings = pd.Series(index=S.index, dtype=float)
    for movie_id, rating in rating_dict.items():
        if movie_id in user_ratings.index:
            user_ratings[movie_id] = rating
    return user_ratings

st.markdown("---")

if st.button("Get Recommendations"):
    if len(user_ratings) > 0:
        user_ratings_matrix = parse_user_ratings(user_ratings)
            
        recommendations = myIBCF(user_ratings_matrix)
        recommended_movie_ids = recommendations.index.tolist()
        
        st.subheader("Your Recommended Movies")
        rec_cols = st.columns(2)
        for idx, movie_id in enumerate(recommended_movie_ids[:10]):
            with rec_cols[idx % 2]:
                movie_idx = movie_id[1:]
                movie_title = get_movie_title(movie_idx)
                st.write(f"### {movie_title}")
                st.image(get_movie_poster(movie_idx), caption=movie_title)
    else:
        st.warning("Please rate at least one movie to get recommendations!")

st.markdown("---")