#Facciamo qualche prova di ricerca semantica fra gli articoli

from sentence_transformers import SentenceTransformer
import math
from pymongo import MongoClient

from heapq import heappop, heappush, heapify


#Connessione a mongodb locale
client_embedding = MongoClient("mongodb://localhost:27017/")
db_embedding = client_embedding['articles_embeddings']
collection_embedding = db_embedding['test_chunks']

#Scelta del modello di encoding
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

#Cosine similarity is the angle between a, b.
#It is computed by (a * b) / ||a * b||
def cosine_similarity(a, b):
	dot = sum(x*y for x,y in zip(a,b))
	norm_a = math.sqrt(sum(x*x for x in a))
	norm_b = math.sqrt(sum(x*x for x in b))
	return dot / (norm_a * norm_b)

#Mantengo un min heap di lunghezza k
#Controllo per ogni similarity se è maggiore del minimo visto finora
#in caso affermativo poppo il minimo e inserisco il nuovo
def find_top_k_chunks(query_vector, k):
	h = []
	heapify(h)
	for chunk in collection_embedding.find():
		similarity = cosine_similarity(chunk['embedding'], query_vector)
		element = (similarity, chunk['embedding'], chunk['text'], chunk['article'])
		if len(h) < k:
			heappush(h, element)
		else:
			if similarity > h[0][0]:
				heappop(h)
				heappush(h, element)
	return sorted(h, key=lambda x: x[0], reverse=True)

query = 'Carlo Nordio'
emb_query = model.encode(query)

best = find_top_k_chunks(emb_query, 10)
best_text = [(x[0], x[2]) for x in best]
print(best_text)




