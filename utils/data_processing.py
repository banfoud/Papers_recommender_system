import pandas as pd

def clean_text(text):

    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

def preprocess_data(data):
    
    data['title'] = data['title'].apply(clean_text)
    data['abstract'] = data['abstract'].apply(clean_text)
    return data