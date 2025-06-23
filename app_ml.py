import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import ast

df = pd.read_csv("predicted_reviews_ml.csv")

st.sidebar.title("Dashboard Options")
sentiment_option = st.sidebar.selectbox("Select Sentiment", ['All', 'Negative', 'Neutral', 'Positive'])

st.title("Sentiment Prediction Dashboard")

st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Predicted_Label', order=['Negative', 'Neutral', 'Positive'], palette='pastel', ax=ax)
st.pyplot(fig)

st.subheader(f"Word Cloud for {sentiment_option}")
if sentiment_option != 'All':
    text = ' '.join(df[df['Predicted_Label'] == sentiment_option]['Review'].dropna())
else:
    text = ' '.join(df['Review'].dropna())

if text.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
else:
    st.write("No text to generate word cloud.")

# 6. Aspect-wise Sentiment Map
if 'Aspects' in df.columns:
    st.subheader("Aspect-wise Sentiment Map")
    aspect_sentiments = []
    for _, row in df.iterrows():
        try:
            aspects = ast.literal_eval(str(row['Aspects']))
            for asp in aspects:
                aspect_sentiments.append((asp.strip(), row['Predicted_Label']))
        except:
            continue

    asp_df = pd.DataFrame(aspect_sentiments, columns=['Aspect', 'Sentiment'])
    if not asp_df.empty:
        top_aspects = asp_df['Aspect'].value_counts().head(10).index
        filtered = asp_df[asp_df['Aspect'].isin(top_aspects)]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=filtered, x='Aspect', hue='Sentiment', order=top_aspects, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.write("No aspect data to display.")

# 7. Sample Table
st.subheader("Review Samples")

columns_to_show = ['Review', 'Predicted_Label']
if 'Aspects' in df.columns:
    columns_to_show.append('Aspects')

if sentiment_option != 'All':
    st.dataframe(df[df['Predicted_Label'] == sentiment_option][columns_to_show])
else:
    st.dataframe(df[columns_to_show])
