# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Load data
df = pd.read_csv("predicted_reviews_transformers.csv")

# 2. Rename + Convert Sentiment Labels
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df.rename(columns={
    'Predicted_Label': 'Predicted',
    'Aspect': 'Aspects'
}, inplace=True)

df['Predicted'] = df['Predicted'].map(label_map)
df['True_Label'] = df['True_Label'].map(label_map)

# 3. Sidebar - Sentiment filter
st.sidebar.title("Dashboard Options")
sentiment_option = st.sidebar.selectbox("Select Sentiment", ['All', 'Negative', 'Neutral', 'Positive'])

# 4. Title
st.title("🧠 Sentiment Prediction Dashboard")

# 5. Sentiment Distribution
st.subheader("Sentiment Distribution")
if 'Predicted' in df.columns and df['Predicted'].notna().any():
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Predicted', order=['Negative', 'Neutral', 'Positive'], palette='pastel', ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ No predicted sentiment data found.")

# 6. Word Cloud
st.subheader(f"Word Cloud for {sentiment_option}")
if sentiment_option != 'All':
    text = ' '.join(df[df['Predicted'] == sentiment_option]['Review'].dropna())
else:
    text = ' '.join(df['Review'].dropna())

if text.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.write("⚠️ No text to generate word cloud.")

# 7. Aspect-wise Sentiment Map
if 'Aspects' in df.columns and df['Aspects'].notna().any():
    st.subheader("Aspect-wise Sentiment Map")

    # Since aspects are just strings, no need to eval
    asp_df = df[['Aspects', 'Predicted']].dropna()
    asp_df.columns = ['Aspect', 'Sentiment']

    if not asp_df.empty:
        top_aspects = asp_df['Aspect'].value_counts().head(10).index
        filtered = asp_df[asp_df['Aspect'].isin(top_aspects)]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=filtered, x='Aspect', hue='Sentiment', order=top_aspects, palette='pastel', ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.write("⚠️ No aspect data to display.")
else:
    st.write("⚠️ No 'Aspects' column found in the data.")

# 8. Review Samples Table
st.subheader("Review Samples")

columns_to_show = ['Review', 'Predicted']
if 'Aspects' in df.columns:
    columns_to_show.append('Aspects')

if sentiment_option != 'All':
    st.dataframe(df[df['Predicted'] == sentiment_option][columns_to_show])
else:
    st.dataframe(df[columns_to_show])
