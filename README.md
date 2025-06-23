# Social Media Computing - Sentiment Analysis Dashboard

This project provides an interactive Streamlit dashboard to visualize sentiment prediction results from three different models in a single interface:

- Machine Learning model
- Deep Learning model
- Transformers (BERT-based) model

---

## Dashboard Features

- Select and compare results from ML, DL, and Transformer models
- Visualize sentiment distribution
- Generate word clouds by sentiment
- Perform aspect-wise sentiment analysis (available for Transformer model)
- View sample predicted reviews for each model and sentiment

---

## How to Run the Dashboard

Run the combined dashboard using the following command:

```bash
streamlit run app.py
```

---

## Requirements

- Streamlit
- Pandas
- Matplotlib
- Seaborn
- WordCloud

```bash
pip install streamlit pandas matplotlib seaborn wordcloud
