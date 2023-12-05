from PIL import Image, ImageEnhance
import shutil
import os
#import deepface
import sys
import argparse
import time
import cv2 as cv
import numpy as np
import pyttsx3




####   Fonction pour dupliquer ler Labels

def duplication_Labels(label_dir_in,label_dir_out,nombre_element):
   # Dupliquer les lignes dans un fichier de sortie
   i=0
   with open(label_dir_out)as file1:
        for line in file1:
           i=i+1

   if i<nombre_element:
        with open(label_dir_in, 'r') as f_entree, open(label_dir_out, 'w') as f_sortie:
            for ligne_entree in f_entree:
        # Écrire la ligne d'origine dans le fichier de sortie
                f_sortie.write(ligne_entree)

        # Dupliquer la ligne 4 fois dans le fichier de sortie
                for _ in range(10):
                    f_sortie.write(ligne_entree)
        pyttsx3.speak("le traitement des labels s'est réaliser sans aucune erreur")
   else:
        pyttsx3.speak("le traitement des labels a déjâ été éffectué")




    ###   fonction pour traiter et dupliquer les images
def traitement_duplication_image(img_dir,add_dir,train_dir,test_dir):
    i=0
    for file in os.listdir(add_dir):
        i=i+1
    if i<=1:
            # phase de traitement
        for fichier in os.listdir(img_dir):
            # chemin_image = os.path.join(img_dir, fichier)
            image=Image.open(os.path.join(img_dir,fichier))
            #rotation=image.rotate(-10)
            #rotation1=image.rotate(10)
            #flip=image.transpose(method=Image.FLIP_LEFT_RIGHT)
            #flip1=image.transpose(method=Image.FLIP_TOP_BOTTOM)
            luminosite=ImageEnhance.Brightness(image).enhance(1.4)
            luminosite1=ImageEnhance.Brightness(image).enhance(0.5)
            couleur=ImageEnhance.Color(image).enhance(2.1)
            couleur1=ImageEnhance.Contrast(image).enhance(1.6)
            forme=ImageEnhance.Color(image).enhance(1.8)
            forme1=ImageEnhance.Color(image).enhance(0.5)

            #on a fini les traitement on duplique l'image en 11 traiter

            image.save(add_dir+"/"+f"{fichier.split('.')[0]}_{0}.jpg")
            #rotation.save(add_dir+"/"+f"{fichier.split('.')[0]}_{1}.jpg")
            #rotation1.save(add_dir+"/"+f"{fichier.split('.')[0]}_{2}.jpg")
            #flip.save(add_dir+"/"+f"{fichier.split('.')[0]}_{3}.jpg")
            #flip1.save(add_dir+"/"+f"{fichier.split('.')[0]}_{4}.jpg")
            luminosite.save(add_dir+"/"+f"{fichier.split('.')[0]}_{1}.jpg")
            luminosite1.save(add_dir+"/"+f"{fichier.split('.')[0]}_{2}.jpg")
            couleur.save(add_dir+"/"+f"{fichier.split('.')[0]}_{3}.jpg")
            couleur1.save(add_dir+"/"+f"{fichier.split('.')[0]}_{4}.jpg")
            forme.save(add_dir+"/"+f"{fichier.split('.')[0]}_{5}.jpg")
            forme1.save(add_dir+"/"+f"{fichier.split('.')[0]}_{6}.jpg")
    

    #Copie des images dupliquées dans les dossiers d'entraînement et de test
        for fichier in os.listdir(add_dir):
            chemin_image_dupliquee = os.path.join(add_dir, fichier)
    
         # Copie dans le dossier d'entraînement
            chemin_image_entrainement = os.path.join(train_dir, fichier)
            shutil.copy(chemin_image_dupliquee, chemin_image_entrainement)
    
        # Copie dans le dossier de test
            chemin_image_test = os.path.join(test_dir, fichier)
            shutil.copy(chemin_image_dupliquee, chemin_image_test)
        pyttsx3.speak("le traitement et la duplication des images s'est bien réalisé")
    else:
        pyttsx3.speak("le traitement a déjâ été effectuer.") 
    i=0
    for file in os.listdir(img_dir):
       i=i+1
    return i



#### la  premiere fonction de duplication de labels  n'est pas bonne car une images est dupliquer 11 fois [0..10] = 11 fois donc un label
#### devrait etre dupliquer en 11 egalement



def label_duplicate_One(label_dir_in,label_dir_out,nombre_element):


    # Dupliquer les lignes dans un fichier de sortie
    with open(label_dir_in, 'r') as f_entree, open(label_dir_out, 'w') as f_sortie:
       

       for ligne_entree in f_entree:

            # Dupliquer la ligne 11 fois dans le fichier de sortie

            # for _ in range(11):
            
            for _ in range(nombre_element):

                f_sortie.write(ligne_entree.rstrip('\n') + '\n')

#----------------------------------------------------------------###

def label_duplicate_Two(label_dir_in,label_dir_out,nombre_element):


    # # Dupliquer les lignes dans un fichier de sortie
    with open(label_dir_in, 'r') as f_entree, open(label_dir_out, 'w') as f_sortie:
       for ligne_entree in f_entree:

            # Écrire la ligne d'origine dans le fichier de sortie
            f_sortie.write(ligne_entree)  ## elle sera compter comme une [premiere ligne donc il reste 10]

            # Dupliquer la ligne 4 fois dans le fichier de sortie

            # for _ in range(10):
            for _ in range(nombre_element-1):

               f_sortie.write(ligne_entree)


###-------------------------------- cependant les autre marche partiellement mais souvent ne conserve pas l'ordre or pour notre projet l'ordre est important  dou celle ci---------###

def label_duplicate_final(label_dir_in, label_dir_out, nombre_element):

   # Dupliquer les lignes dans un fichier de sortie
   i=0
   with open(label_dir_out)as file1:
        for line in file1:
           i=i+1

   if i<nombre_element:
      # Stocker les lignes dupliquées dans une liste
      lignes_dupliquees = []

       # Dupliquer les lignes du fichier d'entrée dans la liste
      with open(label_dir_in, 'r') as f_entree:
          for ligne_entree in f_entree:
              for _ in range(7):
                 lignes_dupliquees.append(ligne_entree.rstrip('\n') + '\n')

       # Écrire les lignes dupliquées dans le fichier de sortie en respectant l'ordre d'origine
      with open(label_dir_out, 'w') as f_sortie:
           for ligne_dupliquee in lignes_dupliquees:
              f_sortie.write(ligne_dupliquee)
               
   else:
        pyttsx3.speak("le traitement des labels a déjâ été éffectué")






#----------------------------------- traitement des images en 100 --------------------------###

from PIL import Image, ImageEnhance, ImageOps

def traitement_duplication_image_two(img_dir, add_dir, train_dir, test_dir):
    i = 0
    for file in os.listdir(add_dir):
        i += 1
    if i <= 1:
        # Phase de traitement et duplication
        for fichier in os.listdir(img_dir):
            image = Image.open(os.path.join(img_dir, fichier))

            traitements = [
                image,
                image.transpose(method=Image.FLIP_LEFT_RIGHT),
                ImageOps.grayscale(image),
                ImageOps.equalize(image),
                ImageEnhance.Brightness(image).enhance(1.2),
                ImageEnhance.Contrast(image).enhance(1.2),
                ImageEnhance.Sharpness(image).enhance(1.2),
            ]

            for j, traitement in enumerate(traitements):
                for k in range(200):
                    # Appliquer d'autres traitements nécessaires sur l'image dupliquée
                    image_dupliquee = traitement.copy()

                    # Sauvegarder l'image dupliquée
                    image_dupliquee.save(add_dir + "/" + f"{fichier.split('.')[0]}_{j*200 + k}.jpg")

                    # Copie dans le dossier d'entraînement
                    chemin_image_entrainement = os.path.join(train_dir, f"{fichier.split('.')[0]}_{j*200 + k}.jpg")
                    shutil.copy(os.path.join(add_dir, f"{fichier.split('.')[0]}_{j*200 + k}.jpg"), chemin_image_entrainement)

                    # Copie dans le dossier de test
                    chemin_image_test = os.path.join(test_dir, f"{fichier.split('.')[0]}_{j*200 + k}.jpg")
                    shutil.copy(os.path.join(add_dir, f"{fichier.split('.')[0]}_{j*200 + k}.jpg"), chemin_image_test)

        pyttsx3.speak("Le traitement et la duplication des images se sont bien réalisés.")
    else:
        pyttsx3.speak("Le traitement a déjà été effectué.")

    i = 0
    for file in os.listdir(img_dir):
        i += 1
    return i







def traitement_duplication_image_final(img_dir, add_dir, train_dir, test_dir):
    i = 0
    for file in os.listdir(add_dir):
        i += 1
    if i <= 1:
        # Phase de traitement et duplication
        for fichier in os.listdir(img_dir):
            image = Image.open(os.path.join(img_dir, fichier))

            # Traitements sur l'image
            rotation = image.rotate(-10)
            rotation1 = image.rotate(10)
            flip = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            flip1 = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            luminosite = ImageEnhance.Brightness(image).enhance(1.4)
            luminosite1 = ImageEnhance.Brightness(image).enhance(0.5)
            couleur = ImageEnhance.Color(image).enhance(2.1)
            couleur1 = ImageEnhance.Contrast(image).enhance(1.6)
            couleur2 = ImageEnhance.Color(image).enhance(1.8)
            couleur3 = ImageEnhance.Color(image).enhance(0.5)

            traitements = [
                image, rotation, rotation1, flip, flip1, luminosite, luminosite1, couleur, couleur1,couleur2, couleur3
            ]


            for j, traitement in enumerate(traitements):
                for k in range(2):
                    # Appliquer d'autres traitements nécessaires sur l'image dupliquée
                    image_dupliquee = traitement.copy()

                    # Sauvegarder l'image dupliquée
                    image_dupliquee.save(add_dir + "/" + f"{fichier.split('.')[0]}_{j * 2 + k}.jpg")

                    # Copie dans le dossier d'entraînement
                    chemin_image_entrainement = os.path.join(train_dir, f"{fichier.split('.')[0]}_{j * 2 + k}.jpg")
                    shutil.copy(os.path.join(add_dir, f"{fichier.split('.')[0]}_{j * 2 + k}.jpg"),
                                chemin_image_entrainement)

                    # Copie dans le dossier de test
                    chemin_image_test = os.path.join(test_dir, f"{fichier.split('.')[0]}_{j * 2 + k}.jpg")
                    shutil.copy(os.path.join(add_dir, f"{fichier.split('.')[0]}_{j * 2 + k}.jpg"), chemin_image_test)

        pyttsx3.speak("Le traitement et la duplication des images se sont bien réalisés.")
    else:
        pyttsx3.speak("Le traitement a déjà été effectué.")

    i = 0
    for file in os.listdir(img_dir):
        i += 1
    return i

