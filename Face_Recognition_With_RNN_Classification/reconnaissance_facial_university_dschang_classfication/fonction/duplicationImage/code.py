import os
import shutil

# Chemin du dossier contenant les images d'origine
chemin_images_origine = "dataImage/"

# Chemin du dossier où les images seront dupliquées
chemin_images_dupliquees = "dataAugmentation/"

# Chemin du dossier où les images d'entraînement seront sauvegardées
chemin_images_entrainement = "trainData/"

# Chemin du dossier où les images de test seront sauvegardées
chemin_images_test = "testData/"

# Copie des images
for fichier in os.listdir(chemin_images_origine):
    chemin_image = os.path.join(chemin_images_origine, fichier)
    
    # Duplique l'image 5 fois
    for i in range(5):
        nouveau_nom = f"{fichier.split('.')[0]}_duplique_{i}.jpg"
        chemin_nouvelle_image = os.path.join(chemin_images_dupliquees, nouveau_nom)
        shutil.copy(chemin_image, chemin_nouvelle_image)

# Copie des images dupliquées dans les dossiers d'entraînement et de test
for fichier in os.listdir(chemin_images_dupliquees):
    chemin_image_dupliquee = os.path.join(chemin_images_dupliquees, fichier)
    
    # Copie dans le dossier d'entraînement
    chemin_image_entrainement = os.path.join(chemin_images_entrainement, fichier)
    shutil.copy(chemin_image_dupliquee, chemin_image_entrainement)
    
    # Copie dans le dossier de test
    chemin_image_test = os.path.join(chemin_images_test, fichier)
    shutil.copy(chemin_image_dupliquee, chemin_image_test)
