from transformers import pipeline
import pandas as pd

# Sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis",  model='distilbert-base-uncased-finetuned-sst-2-english')

def get_mood_profile(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    mood = {}

    if label == 'POSITIVE':
        mood['valence'] = 0.8 + 0.2 * score                   # 0.8–1.0
        mood['energy'] = 0.6 + 0.4 * score                    # 0.6–1.0
        mood['danceability'] = 0.5 + 0.5 * score              # 0.5–1.0
        mood['tempo'] = int(100 + 50 * score)                 # 100–150 bpm

    elif label == 'NEGATIVE':
        mood['valence'] = max(0.0, 0.2 - 0.2 * score)         # 0.0–0.2
        mood['energy'] = max(0.0, 0.4 - 0.3 * score)          # 0.1–0.4
        mood['danceability'] = max(0.0, 0.5 - 0.3 * score)    # 0.2–0.5
        mood['tempo'] = int(90 - 30 * score)                  # 60–90 bpm

    return mood

def recommend_songs(user_text, df, excluded_genres=None):
    """
    user_text: text input for mood analysis
    df: DataFrame of songs with mood metrics
    excluded_genres: list of genres to exclude (optional)
    """

    mood = get_mood_profile(user_text)
    df = df.copy()

    # Optional genre exclusion
    if excluded_genres:
        df = df[~df['track_genre'].isin(excluded_genres)]

    # Mood matching logic
    df['distance'] = (
        (df['valence'] - mood['valence']) ** 2 +
        (df['energy'] - mood['energy']) ** 2 +
        (df['danceability'] - mood['danceability']) ** 2 +
        ((df['tempo'] - mood['tempo']) / 100) ** 2
    ) ** 0.5

    matched = df.sort_values(by='distance').head(50)

    return matched[['track_name', 'artists', 'track_genre', 'valence', 'energy', 'danceability', 'tempo']]
