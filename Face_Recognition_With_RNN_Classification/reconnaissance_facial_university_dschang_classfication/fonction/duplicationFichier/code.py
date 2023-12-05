import shutil

# Chemin des fichiers d'entrée et de sortie
fichier_entree = "labels/labels.txt"
fichier_sortie = "sortie/data.txt"
dossier_entrainement = "trainData/"
dossier_test = "testData/"

# Dupliquer les lignes dans un fichier de sortie
with open(fichier_entree, 'r') as f_entree, open(fichier_sortie, 'w') as f_sortie:
    for ligne_entree in f_entree:
        # Écrire la ligne d'origine dans le fichier de sortie
        f_sortie.write(ligne_entree)

        # Dupliquer la ligne 4 fois dans le fichier de sortie
        for _ in range(4):
            f_sortie.write(ligne_entree)

# Copier le fichier de sortie dans le dossier d'entraînement et de test
shutil.copy(fichier_sortie, dossier_entrainement)
shutil.copy(fichier_sortie, dossier_test)
