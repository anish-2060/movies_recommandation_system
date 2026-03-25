import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

# ---------------- LOAD DATA ---------------- #
@st.cache_resource
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    
    # Transform (NO fit_transform)
    vectors = vectorizer.transform(movies['tags'])
    
    # Build model (do NOT load pickle model)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)
    
    return movies, model, vectors

movies, model, vectors = load_data()

# ---------------- MOVIE LIST ---------------- #
movie_list = movies['title'].values

selected_movie = st.selectbox("Select a movie", movie_list)

# ---------------- RECOMMEND FUNCTION ---------------- #
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    
    distances, indices = model.kneighbors(
        vectors[index], 
        n_neighbors=6
    )

    recommended_movies = []
    for i in indices[0][1:]:
        recommended_movies.append(movies.iloc[i].title)
    
    return recommended_movies

# ---------------- BUTTON ---------------- #
if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        recommendations = recommend(selected_movie)
        
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)