import cv2

def capture_image():
    camera = cv2.VideoCapture(1)  # Ouvre la webcam (index 0)
    while True:
        ret, frame = camera.read()  # Lit une image depuis la webcam
        cv2.imshow("Webcam", frame)  # Affiche l'image dans une fenêtre

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Si la touche 'c' est pressée
            cv2.imwrite("capture/image_capturee.jpg", frame)  # Enregistre l'image capturée
            break

    camera.release()  # Libère la webcam
    cv2.destroyAllWindows()  # Ferme toutes les fenêtres

capture_image()
