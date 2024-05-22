import streamlit as st
import pandas as pd
from preprocess_data import preprocess_text

TITLE_TEXT = f"IMDB reviews"
DESCRIPTION_TEXT = f"Esta é uma aplicação para o trabalho de NLP. Utilizamos a base de dados de reviews do IMDb com 50.000 comentários entre positivos e negativos (a base está balanceada). Por meio desta interface é possível visualizar como os exemplos da nossa base de teste foram classificados com um BERT treinado para esta task."

st.title(TITLE_TEXT)
st.write(DESCRIPTION_TEXT)

df = pd.read_csv('data/imdb_reviews.csv')
df = df.head()
df['preprocessed_review'] = df['review'].copy()
df['preprocessed_review'] = df['preprocessed_review'].apply(preprocess_text)
cols = ['review','preprocessed_review','sentiment']
st.table(df[cols])