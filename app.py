import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-saved models and data
tfidf = joblib.load('tfidf_model.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')
df = pd.read_csv('netflix_data_processed.csv')

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie/show that matches the title
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return ["Title not found. Please check your spelling or try another title."]
    
    # Get the pairwise similarity scores for all shows/movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort shows/movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of the most similar shows/movies
    sim_indices = [i[0] for i in sim_scores[1:11]]  # Top 10 recommendations
    
    # Return the titles of the most similar shows/movies
    return df['title'].iloc[sim_indices].tolist()

# Streamlit app interface
st.title('Netflix Movie/Show Recommendation System')
st.write('Enter a movie or TV show title to get recommendations!')

# Input box for user to enter a title
user_input = st.text_input('Enter the title of a movie or TV show:')

# Button to trigger recommendations
if st.button('Get Recommendations'):
    if user_input:
        recommendations = get_recommendations(user_input)
        if recommendations:
            st.write('Top 10 recommendations:')
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write('No recommendations found. Please try a different title.')
    else:
        st.write('Please enter a title to get recommendations.')