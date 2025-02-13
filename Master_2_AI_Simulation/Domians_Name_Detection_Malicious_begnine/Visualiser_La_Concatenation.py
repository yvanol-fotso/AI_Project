import pandas as pd

# Création de deux DataFrames avec des colonnes identiques
df1 = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
df2 = pd.DataFrame({'feature_1': [7, 8, 9], 'feature_2': [10, 11, 12]})

# Concaténation verticale (les données s'empilent)
merged_data = pd.concat([df1, df2], axis=0, ignore_index=True)

print(merged_data)


print("\n")


# Création de deux DataFrames avec des colonnes differents
df3 = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
df4 = pd.DataFrame({'feature_3': [7, 8, 9], 'feature_4': [10, 11, 12]})


# Création de deux DataFrames avec des colonnes differents
df5 = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
df6 = pd.DataFrame({'feature_1': [7, 8, 9], 'feature_2': [10, 11, 12],'feature_3': [00, 11, 22]})

# Concaténation verticale (les données se juxtaposent)
#merged_data_2= pd.concat([df3, df4], axis=1, ignore_index=True) # empeche de voir le nom des colonne

merged_data_2= pd.concat([df3, df4], axis=1)
merged_data_3= pd.concat([df3, df4], axis=0)


#####3 TRES IMPORTANT A BIEN COGITER SUR LA SORTIE CONCLUSION IL FAUT PLUS CONCATENERT SUR AXIS = 0(CAS EJA EVITE E DUPLIQUER LES COLONE DE MEME NONM)
merged_data_4= pd.concat([df5, df6], axis=1)
merged_data_5= pd.concat([df5, df6], axis=0)

print(merged_data_2)
print("\n")
print(merged_data_3)

print("\n")
print(merged_data_4)
print("\n")
print(merged_data_5)



