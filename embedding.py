"""Idea: Dividere gli articoli in chunks, tenendo traccia per ciascun chunk di quale articolo fa parte. 
Nel momento in cui un chunk viene selezionato si ritornano tutti i restanti chunk di quel documento."""

from sentence_transformers import SentenceTransformer
import math
from pymongo import MongoClient

#Connessione a MongoDB in locale (IP di WSL)
client_articles = MongoClient("mongodb://localhost:27017/")
db_articles = client_articles['articles']
collection_articles = db_articles['ilpost']

client_embedding = MongoClient("mongodb://localhost:27017/")
db_embedding = client_embedding['articles_embeddings']
collection_embedding = db_embedding['chunks']

#Dato un documento ritorna una lista di chunks del documento di lunghezza l
#Oss: capire per la dimensione dei chunks, inoltre sarebbe bello provare anche 
#altro oltre a vector search (keyword based per esempio)
def make_chunks(doc, l):
	chunks = [doc[i:i+l] for i in range(0, len(doc), l)]
	return chunks


model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

i = 0
docs = collection_articles.count_documents({})

#Per tenere traccia di quale articolo fanno parte gli embedding associo ad ogni articolo intero un numero progressivo
#Che salvo in ciascun documento contenente embedding di parti di quell'articolo
progressive_embedding_id = collection_embedding.count_documents({})

for article in collection_articles.find():
	i = i + 1
	progressive_embedding_id = progressive_embedding_id + 1

	print(f'Processing article {i}/{docs}...')

	chunks = make_chunks(article['title'] + article['body'], 100)

	print(f'Encoding Chunks...')
	embeddings = model.encode(chunks)

	print(f'Saving embeddings...')
	for embedding, chunk in zip(embeddings, chunks):
		collection_embedding.insert_one({'article': progressive_embedding_id, 'text': chunk, 'embedding': embedding.tolist()})



"""Per il salvataggio degli embedding userei brutalmente mongoDB:
	{
		"encoding" : [ ... ] //Vettore
		"chunk" : ...testo...
	}
"""

