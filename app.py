import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors
import requests
import os

# ---------------- GOOGLE DRIVE DOWNLOAD FUNCTION ---------------- #
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ---------------- DOWNLOAD DATA IF NOT PRESENT ---------------- #
if not os.path.exists("movies.pkl"):
    file_id = "1_qPy35cbwcntbvGvsjaBZ7XTlaNkpTh2"   # your file ID
    download_file_from_google_drive(file_id, "movies.pkl")

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# ---------------- LOAD DATA ---------------- #
@st.cache_resource
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # Transform (NO fit_transform)
    vectors = vectorizer.transform(movies['tags'])

    # Build NearestNeighbors model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(vectors)

    return movies, model, vectors

movies, model, vectors = load_data()

# ---------------- DROPDOWN ---------------- #
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
            st.write("👉", movie)