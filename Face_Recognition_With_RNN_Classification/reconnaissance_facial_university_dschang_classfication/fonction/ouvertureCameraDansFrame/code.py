# 1. Importez les modules nécessaires :
import tkinter as tk
import cv2
from PIL import ImageTk, Image


# 2. Créez une fenêtre principale et une frame à l'intérieur :
root = tk.Tk()

frame = tk.Frame(root, width=640, height=480)
frame.pack()


# 3. Créez une fonction pour afficher les images de la caméra dans cette frame :

def afficher_camera():
    cap = cv2.VideoCapture(1)  # Utilisez 1 si vous avez plusieurs caméras
    
    _, frame_camera = cap.read()
    cv2.imshow("Webcam", frame_camera)  # Affiche l'image dans une fenêtre


    image = Image.fromarray(frame_camera)
    image = ImageTk.PhotoImage(image)

    label_camera = tk.Label(frame, image=image)
    label_camera.image = image
    label_camera.pack()

    # root.after(10, afficher_camera)  # Rafraîchit l'affichage toutes les 10 millisecondes

afficher_camera()  # Appel initial de la fonction


# 4. Exécutez la boucle principale de la fenêtre :

root.mainloop()

