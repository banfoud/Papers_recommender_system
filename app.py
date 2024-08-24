import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_json('C:\\Users\\Brahim Anfoud\\Desktop\\Arxiv_recommender_system\\arxiv-metadata-oai-snapshot.json', lines = True)
    data['update_date'] = pd.to_datetime(data['update_date'])
    filtered_data = data[data['update_date'] > '2019-12-31']
    filtered_data['text'] = filtered_data['title'] + ' ' + filtered_data['abstract']
    return filtered_data

data = load_data()

@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

@st.cache(allow_output_mutation=True)
def compute_embeddings(data):
    embeddings = model.encode(data['text'].tolist(), convert_to_tensor=True)
    return embeddings

embeddings = compute_embeddings(data)

st.title("Recommender System for Scientific Articles")

tab1, tab2 = st.tabs(["🔍 Article Recommendations", "📊 Data Exploration"])

with tab1:
    st.header("Find Similar Articles")
    input_title = st.text_input("Enter the title of your article")
    input_abstract = st.text_area("Enter the abstract of your article")

    if st.button("Recommend Articles"):
        if input_title and input_abstract:
            user_input = input_title + ' ' + input_abstract
            user_embedding = model.encode([user_input], convert_to_tensor=True)
            cosine_sim = cosine_similarity(user_embedding.cpu().numpy(), embeddings.cpu().numpy())
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_k = 8
            article_indices = [i[0] for i in sim_scores[:top_k]]

            st.write(f"Top {top_k} recommended articles:")
            for idx in article_indices:
                st.write(f"**Title:** {data['title'].iloc[idx]}")
                st.write(f"**Abstract:** {data['abstract'].iloc[idx]}")
                st.write(f"**Update Date:** {data['update_date'].iloc[idx]}")
                st.write("---")
        else:
            st.warning("Please enter both title and abstract.")

with tab2:
    st.header("Data Exploration")

    # Distribution de la longueur des titres
    st.subheader("Distribution de la longueur des titres")
    data['title_length'] = data['title'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['title_length'], bins=30, kde=True)
    st.pyplot(plt)

    # Distribution de la longueur des résumés
    st.subheader("Distribution de la longueur des résumés")
    data['abstract_length'] = data['abstract'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['abstract_length'], bins=30, kde=True)
    st.pyplot(plt)

    # Nuage de mots pour les titres
    st.subheader("Nuage de mots pour les titres")
    wordcloud_title = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['title']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_title, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Nuage de mots pour les résumés
    st.subheader("Nuage de mots pour les résumés")
    wordcloud_abstract = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['abstract']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_abstract, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

