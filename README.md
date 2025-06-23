# Social Media Computing - Sentiment Analysis Dashboard

This project provides **interactive dashboards** to visualize sentiment prediction results from three different models:

- `app_ml.py` — Machine Learning model dashboard  
- `app_dl.py` — Deep Learning model dashboard  
- `app_transformers.py` — Transformers (BERT-based) model dashboard

Each dashboard shows:
- Sentiment distribution
- Word cloud by sentiment
- Aspect-wise sentiment analysis(app_transformers.py only)
- Sample predicted reviews

---

## How to Run Each Dashboard

- `app_ml.py` — streamlit run app_ml.py
- `app_dl.py` — streamlit run app_dl.py
- `app_transformers.py` — streamlit run app_transformers.py
- 
### Requirements:
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- WordCloud

You can install them with:

```bash
pip install streamlit pandas matplotlib seaborn wordcloud
