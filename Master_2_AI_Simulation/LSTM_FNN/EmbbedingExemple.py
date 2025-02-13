#Convertion des tokens en vecteurs d'embeddings en utilisant le modèle Word2Vec :

import gensim.downloader as api

# Téléchargement du modèle Word2Vec pré-entraîné
model = api.load("frWiki_no_phrase_no_postag_300d")


# Exemple de tokens
tokens_feature1 = [1, 2, 3, 4, 5]
tokens_feature2 = [1, 2, 6, 7, 3, 4, 5]

# Convertir les tokens en embeddings
embeddings_feature1 = [model[word] for word in tokens_feature1]
embeddings_feature2 = [model[word] for word in tokens_feature2]

print("Embeddings feature 1:", embeddings_feature1)
print("Embeddings feature 2:", embeddings_feature2)
