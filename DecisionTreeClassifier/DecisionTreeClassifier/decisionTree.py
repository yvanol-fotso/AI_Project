import pandas as pd
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import joblib
from sklearn import metrics
import matplotlib.pyplot as plt





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

# Créer un arbre de décision / build model notre model ici c'est le clf

# clf = DecisionTreeClassifier() #ici on dessine tout l'arbre

# controle de la profondeur de l'arbre

clf = DecisionTreeClassifier(max_depth=5)

#Le paramètre ' min_samples_leaf= entier' donne le nombre
# minimal d’échantillons dans un nœud feuille.  donc on peut changer le max_depth par celuici

clf.fit(X_train, y_train)


clf.fit(X_train, y_train)
joblib.dump(clf, "yvapro_test")  #store the trained model


predictions = clf.predict(X_test)
print(metrics.classification_report(predictions, y_test))



print("\n")


# Utiliser l'ensemble de test pour évaluer les performances du modèle
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))


# Precision: tp/ (tp + fp)
# Recall   : tp / (tp + fn)
# F1-score : 2 * precision * recall / (precision + recall)
# Suppor   : Number of occurrences of each class in y_true



# # Tracer l'arbre de décision
plt.figure(figsize=(15, 10))
# plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names(), class_names=clf.classes_)
plot_tree(clf, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=clf.classes_)

plt.show()


# new_data = pd.read_csv('newSymptom.csv')

# new_data = "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."
new_data = "my temperature is high and i'm burning"



# Prétraiter la nouvelle donnée avec TfidfVectorizer
new_data_vectorized = vectorizer.transform([new_data])

# Faire une prédiction avec le modèle entraîné
prediction = clf.predict(new_data_vectorized)


# Afficher la prédiction
print("La prédiction est / il est probable que vous souffrez de :", prediction)
