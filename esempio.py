# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


# Carica il dataset
dataset = pd.read_csv("Tiny Eco Wonderflow3.csv", sep=";", on_bad_lines="skip")  # Prova con ";" come separatore

dataset.columns = dataset.columns.str.strip()

# Estrai la colonna 'text'
texts = dataset['text'].dropna()
texts = texts[texts.str.strip() != '']

# Rimuovi gli spazi all'inizio e alla fine del testo, e le nuove righe extra
texts_cleaned = texts.str.replace(r'\s+', ' ', regex=True)  # Rimuove spazi extra (compresi \n e \t)
texts = texts_cleaned.str.strip()  # Rimuove gli spazi ai bordi (inizio e fine del testo)

# Funzione per ottenere il sentiment per ogni testo
def get_sentiment(text):
    # Tokenizza il testo
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Invia il testo attraverso il modello
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ottieni le probabilità di sentiment (5 classi da 0 a 4)
    sentiment_scores = outputs.logits
    sentiment = torch.argmax(sentiment_scores, dim=1).item()  # Ottieni la classe con la probabilità più alta

    sentiment = sentiment + 1
    
    return {'text': text, 'sentiment': sentiment, 'trueSentiment': dataset['feedbackRating']}

# Applica la funzione di sentiment analysis a tutti i testi
sentiments = texts.apply(get_sentiment)

print(sentiments.head(20))