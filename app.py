import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import ast

# Sidebar - Model and Sentiment Selection
st.sidebar.title("Dashboard Options")

model_option = st.sidebar.selectbox("Select Model", ['ML', 'DL', 'Transformer'])
sentiment_option = st.sidebar.selectbox("Select Sentiment", ['All', 'Negative', 'Neutral', 'Positive'])

# Load Data Based on Model
if model_option == 'ML':
    df = pd.read_csv("predicted_reviews_ml.csv")
    sentiment_column = 'Predicted_Label'
elif model_option == 'DL':
    df = pd.read_csv("predicted_reviews_dl.csv")
    sentiment_column = 'Predicted_Label'
elif model_option == 'Transformer':
    df = pd.read_csv("predicted_reviews_transformers.csv")
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df.rename(columns={'Predicted_Label': 'Predicted', 'Aspect': 'Aspects'}, inplace=True)
    df['Predicted'] = df['Predicted'].map(label_map)
    df['True_Label'] = df['True_Label'].map(label_map)
    sentiment_column = 'Predicted'

# Title
st.title(f"Sentiment Prediction Dashboard - {model_option} Model")

# Sentiment Distribution
st.subheader("Sentiment Distribution")
if sentiment_column in df.columns and df[sentiment_column].notna().any():
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=sentiment_column, order=['Negative', 'Neutral', 'Positive'], palette='pastel', ax=ax)
    st.pyplot(fig)

# Word Cloud
st.subheader(f"Word Cloud for {sentiment_option}")
if sentiment_option != 'All':
    text = ' '.join(df[df[sentiment_column] == sentiment_option]['Review'].dropna())
else:
    text = ' '.join(df['Review'].dropna())

if text.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

# Aspect-wise Sentiment Map
if 'Aspects' in df.columns and df['Aspects'].notna().any():
    st.subheader("Aspect-wise Sentiment Map")
    aspect_sentiments = []

    if model_option in ['ML', 'DL']:
        for _, row in df.iterrows():
            try:
                aspects = ast.literal_eval(str(row['Aspects']))
                for asp in aspects:
                    aspect_sentiments.append((asp.strip(), row[sentiment_column]))
            except:
                continue
    else:  # Transformer
        aspect_sentiments = list(zip(df['Aspects'], df[sentiment_column]))

    asp_df = pd.DataFrame(aspect_sentiments, columns=['Aspect', 'Sentiment'])
    if not asp_df.empty:
        top_aspects = asp_df['Aspect'].value_counts().head(10).index
        filtered = asp_df[asp_df['Aspect'].isin(top_aspects)]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=filtered, x='Aspect', hue='Sentiment', order=top_aspects, palette='pastel', ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# Review Samples Table
st.subheader("Review Samples")

columns_to_show = ['Review', sentiment_column]
if 'Aspects' in df.columns:
    columns_to_show.append('Aspects')

if sentiment_option != 'All':
    st.dataframe(df[df[sentiment_column] == sentiment_option][columns_to_show])
else:
    st.dataframe(df[columns_to_show])
