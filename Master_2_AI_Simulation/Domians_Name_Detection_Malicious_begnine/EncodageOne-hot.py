import pandas as pd

# Création d'un dataframe de données d'exemple
data = pd.DataFrame({'attribut_catégorique': ['A', 'B', 'A', 'C', 'B']})

# Encodage des attributs catégoriques
encoded_categorical_data = pd.get_dummies(data['attribut_catégorique'])

print(encoded_categorical_data)