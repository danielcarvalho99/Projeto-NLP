import streamlit as st
import pandas as pd
from preprocess_data import preprocess_text,get_stopwords
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from wordnet import wordnet_pipeline

dataset = load_dataset('AndreMitri/rotten_tomatos')

dataframes = {}
for split in dataset.keys():
    # Convert the dataset split to a pandas DataFrame
    df = dataset[split].to_pandas()
    dataframes[split] = df


MODEL_PATH = 'danielcd99/BERT_imdb'

def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.model_max_length = 200

    pipe=pipeline(
    "text-classification",
    model=MODEL_PATH
    )
    return pipe

pipe = load_pipeline()


TITLE_TEXT = f"IMDB reviews"
DESCRIPTION_TEXT = f"Esta é uma aplicação para o trabalho de NLP. Utilizamos a base de dados de reviews do IMDb com 50.000 comentários entre positivos e negativos (a base está balanceada). Por meio desta interface é possível visualizar como os exemplos da nossa base de teste foram classificados com um BERT treinado para esta task."

st.title(TITLE_TEXT)
st.write(DESCRIPTION_TEXT)

if st.button('Encontre exemplos!'):
    df = df.sample(5)
    get_stopwords()
    df['preprocessed_review'] = df['review'].copy()
    df['preprocessed_review'] = df['preprocessed_review'].apply(preprocess_text)
    
    predictions = []
    for review in df['preprocessed_review']:
        try:
            label = pipe(review)[0]['label']
        except:
            st.error("Ocorreu um erro de carregamento, tente novamente!")
        
        if label == 'LABEL_0':
            predictions.append('Negative')
        else:
            predictions.append('Positive')

    df['bert_results'] = predictions
    df['wordnet_results'] = wordnet_pipeline(df, 'preprocessed_review')

    cols = ['review','sentiment', 'bert_results', 'wordnet_results']    

    st.table(df[cols])

