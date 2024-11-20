# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


# Carica il primo dataset
dataset1 = pd.read_csv("Tiny Eco Wonderflow3.csv", sep=";", on_bad_lines="skip")  # Prova con ";" come separatore
dataset1.columns = dataset1.columns.str.strip()

# Estrai la colonna 'text' dal primo dataset
texts1 = dataset1['text'].dropna()
texts1 = texts1[texts1.str.strip() != '']

# Rimuovi gli spazi all'inizio e alla fine del testo, e le nuove righe extra nel primo dataset
texts1_cleaned = texts1.str.replace(r'\s+', ' ', regex=True)  # Rimuove spazi extra (compresi \n e \t)
texts1 = texts1_cleaned.str.strip()  # Rimuove gli spazi ai bordi

# Carica il secondo dataset
dataset2 = pd.read_csv("Tiny Eco Digimind.csv", sep=";", on_bad_lines="skip")  # Prova con ";" come separatore
dataset2.columns = dataset2.columns.str.strip()

# Estrai e concatena 'Title' e 'Detail' nel secondo dataset
texts2 = dataset2['Title'].fillna('') + ' ' + dataset2['Detail'].fillna('')
texts2 = texts2[texts2.str.strip() != '']

# Unisci i due datasets di testo (puoi usare un'unica variabile per combinarli)
texts= pd.concat([texts1, texts2], ignore_index=True)


# Funzione per ottenere il sentiment per ogni testo
def get_sentiment(text, true_sentiment):
    # Tokenizza il testo
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Invia il testo attraverso il modello
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ottieni le probabilità di sentiment (5 classi da 0 a 4)
    sentiment_scores = outputs.logits
    sentiment = torch.argmax(sentiment_scores, dim=1).item()  # Ottieni la classe con la probabilità più alta
    
    sentiment = sentiment + 1  # Mappa da 0-4 a 1-5
    
    return {'text': text, 'sentiment': sentiment, 'trueSentiment': true_sentiment}

# Applica la funzione di sentiment analysis a tutti i testi
#sentiments = [get_sentiment(text, true_sentiment) for text, true_sentiment in zip(texts, dataset['feedbackRating'])]

#print(sentiments)

# Converte i risultati in un DataFrame
#sentiments_df = pd.DataFrame(sentiments)

# Calcola l'accuratezza: confronta sentiment e trueSentiment
#correct_predictions = (sentiments_df['sentiment'] == sentiments_df['trueSentiment']).sum()
#accuracy = correct_predictions / len(sentiments_df) * 100

# Calcola l'accuracy off-by-one: quando la differenza tra sentiment e trueSentiment è 1
#accuracy_off_by_one = (abs(sentiments_df['trueSentiment'] - sentiments_df['sentiment']) <= 1).sum()
#accuracy_off_by_one_percent = accuracy_off_by_one / len(sentiments_df) * 100


# Stampa l'accuratezza e l'accuracy-1
#print(f"Accuracy: {accuracy:.2f}%")
#print(f"Accuracy off-by-one: {accuracy_off_by_one_percent:.2f}%")

#accuracy 30%, off by one 65.4%