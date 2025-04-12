import streamlit as st
import pandas as pd
from utils.mood_matcher import recommend_songs
import os

os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"

def main():
    # Load dataset
    df = pd.read_csv("data/cleaned_spotify_dataset.csv")

    # UI
    st.title("🎧 Moodi - Mood Based Playlist Generator")
    st.markdown("Type how you're feeling, and get a custom playlist!")

    # User input
    user_input = st.text_input("How are you feeling today?")

    # Genre filtering
    available_genres = sorted(df['track_genre'].dropna().unique())
    excluded_genres = st.multiselect(
        "Exclude genres you're not in the mood for:",
        available_genres
    )

    if user_input:
        st.subheader("Here are your songs 🎶")
        recs = recommend_songs(user_input, df, excluded_genres=excluded_genres)
        for i, row in recs.iterrows():
            st.write(f"**{row['track_name']}** by {row['artists']} (Genre: {row['track_genre']}, Valence: {row['valence']:.2f}, Energy: {row['energy']:.2f})")

if __name__ == "__main__":
    main()
