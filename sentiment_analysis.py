import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# ------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    tweet = re.sub(r'@\w+|#', '', tweet)
    tweet = re.sub(r'[^A-Za-z0-9 ]+', ' ', tweet)
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.split() if word.lower() not in stop_words])
    return tweet

# ------------------------------------------
# EMOJI SENTIMENT LABELS
# ------------------------------------------
def emoji_label(score):
    if score > 0.05:
        return "😊 Positive"
    elif score < -0.05:
        return "😞 Negative"
    else:
        return "😐 Neutral"


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="Tweet Sentiment Analysis", layout="wide")
st.title("💬 Tweet Sentiment Analysis")
st.write("Analyze multiple tweets and visualize their sentiment with charts, scores, and emojis.")

tweets_input = st.text_area("Enter tweets (one per line):", height=150)

if st.button("Analyze"):
    if tweets_input.strip():

        tweets = tweets_input.split('\n')
        df = pd.DataFrame({'tweets': tweets})

        # Preprocess
        df['cleaned_tweet'] = df['tweets'].apply(preprocess_tweet)

        # Sentiment Analysis
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['cleaned_tweet'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

        # Labels
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
        )
        df['emoji_label'] = df['sentiment_score'].apply(emoji_label)

        # Display Results
        st.subheader("📊 Sentiment Results")
        st.dataframe(df[['tweets', 'emoji_label', 'sentiment_score']])

        # ------------------------------------------
        # SCORE GAUGE METER (Average Sentiment)
        # ------------------------------------------
        avg_score = df['sentiment_score'].mean()

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={'text': "Overall Sentiment Score"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [-1, -0.05], 'color': 'red'},
                    {'range': [-0.05, 0.05], 'color': 'gray'},
                    {'range': [0.05, 1], 'color': 'green'}
                ]
            }
        ))
        st.subheader("📈 Sentiment Score Meter")
        st.plotly_chart(gauge, use_container_width=True)

        # ------------------------------------------
        # PIE CHART
        # ------------------------------------------
        st.subheader("🥧 Sentiment Distribution (Pie Chart)")
        sentiment_counts = df['sentiment_label'].value_counts()
        pie_chart = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "green", 
                "Negative": "red",
                "Neutral": "gray"
            },
            title="Pie Chart of Sentiment"
        )
        st.plotly_chart(pie_chart, use_container_width=True)

        # ------------------------------------------
        # BAR CHART
        # ------------------------------------------
        st.subheader("📊 Sentiment Distribution (Bar Chart)")
        bar_chart = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "green",
                "Negative": "red",
                "Neutral": "gray"
            }
        )
        st.plotly_chart(bar_chart, use_container_width=True)

    else:
        st.warning("⚠ Please enter at least one tweet!")
