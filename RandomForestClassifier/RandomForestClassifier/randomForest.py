import pandas as pd

from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
import graphviz

from sklearn import metrics
import joblib

from IPython.display import Image

import numpy as np

import pydotplus

import matplotlib.pyplot as plt #toujuors pour voir l'affichage




# Charger les données à partir du fichier CSV
data = pd.read_csv('Symptom2Disease.csv')

# Séparer les caractéristiques (symptômes) et les étiquettes
X = data['text'].values
y = data['label'].values

# Prétraitement des données
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build model
model = RandomForestClassifier(n_estimators = 100, max_depth = 5)
model.fit(X_train, y_train)

joblib.dump(model, "yvapro_test")  #store the trained model




predictions = model.predict(X_test)
print(metrics.classification_report(predictions, y_test))


print("\n")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))



# Tracer l'arbre de décision
estimator = model.estimators_[0]
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = vectorizer.get_feature_names_out(),
                class_names = model.classes_,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)



#test de dessinage du graphe avec plot

# # Tracer l'arbre de décision
# plt.figure(figsize=(10, 10))

# #plot_tree(model.estimators_[0], filled=True) #sa marche mais le accuracy n'est pas bon
# plot_tree(model.estimators_[80], filled=True,class_names=model.classes_)

# plt.show()




# Exporter le graphique de l'arbre de décision
# dot_data = export_graphviz(model.estimators_[0], 
#                            out_file=None, 
#                            feature_names=vectorizer.get_feature_names_out(),  
#                            class_names=np.unique(y_train),  
#                            filled=True, rounded=True,  
#                            special_characters=True)  
# graph = pydotplus.graph_from_dot_data(dot_data)  
# Image(graph.create_png())