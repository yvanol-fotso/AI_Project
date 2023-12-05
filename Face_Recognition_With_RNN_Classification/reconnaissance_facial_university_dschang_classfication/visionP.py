import os
import sys
import argparse
import time
import cv2 as cv
import numpy as np

faces=cv.CascadeClassifier('algorithme/haarcascade_frontalface_default.xml')
# eye=cv.CascadeClassifier('algorithme/haarcascade_eye.xml')

#creons une class image pour manipuler nos image
class Capture:
    #constructeur d'image
    def __init__(self,img):
        self.ima=cv.imread(img,cv.IMREAD_UNCHANGED)
        if self.ima is None:
         sys.exit("erreur lors du chargement de l'image")
             
    #methode show() pour afficher une image

    def show(self):
        cv.imshow("image",self.ima)
        print("dimension:",self.ima.shape)
        cv.waitKey()


    ##methode pour redimensionner une image resize()
    def resize(self):
        largeur=300
        hauteur=300
        dim=(largeur,hauteur)
        self.ima=cv.resize(self.ima,dim)
            
    #methode compare() pour compare les image
    def compare(self,image):
        im1=cv.cvtColor(self.ima,cv.COLOR_BGR2GRAY)
        im2=cv.cvtColor(image.ima,cv.COLOR_BGR2GRAY)
        def erreur(im1,im2):
         h,w=im1.shape
         diff =cv.subtract(im1,im2)
         err=np.sum(diff**2)
         mse=err/(float(h*w))
         return mse
        diff=erreur(im1,im2)
        print("voici la difference",diff)
        # i2=image.face_detect()
        if diff<=1.2:
            print("welcome becky")
        else:
            print("image differentes")
    #methode face_detect() pour detecter la face sur une image
    def face_detect(self):
        gray=cv.cvtColor(self.ima,cv.COLOR_BGR2RGBA)
        face=faces.detectMultiScale(gray,1.3,5)
        print("found {0} faces".format(len(face))) 
        return face
    #methode draw_face() pour tracer la face et les yeux
    def draw_face(self):
        gray=cv.cvtColor(self.ima,cv.COLOR_BGR2RGBA)
        face=self.face_detect()
        for (x,y,w,h) in face:
         cv.rectangle(self.ima,(x,y),(x+w,y+h),(0,255,0),2)
         eyes_gray=gray[y:y+h,x:x+w]
         eyes_color=self.ima[y:y+h,x:x+w]
         eyes=eye.detectMultiScale(eyes_gray)
         for (ex,ey,ew,eh) in eyes:
          cv.rectangle(eyes_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #methode get_pic_from_strem() pour obtenir une image a 
          # partir de la camera
    def get_pic_from_strem(n):

     mage=cv.VideoCapture(n)
     while True:
      ret, frame=mage.read()
      gray=cv.cvtColor(frame,cv.COLOR_BGR2RGBA)
      face=faces.detectMultiScale(gray,1.3,5) 
      if len(face) ==1:
       cv.imwrite("captureAutomatique/capture.jpg",frame)
       break 


    # #dessiner les rectangle autour des faces
      for (x,y,w,h) in face:
       cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
       eyes_gray=gray[y:y+h,x:x+w]
       eyes_color=frame[y:y+h,x:x+w]
       eyes=eye.detectMultiScale(eyes_gray,1.3,5)
       for (ex,ey,ew,eh) in eyes:
        cv.rectangle(eyes_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
      cv.imshow("Capture Automatique",frame)
      if cv.waitKey(20) == ord('q'):
       cv.destroyWindow("Capture Automatique")
       break  
     return "captureAutomatique/capture.jpg"
    
# im1=image(image.get_pic_from_strem(0))
# im1.resize()
# im.resize()
# im1.draw_face()
# im1.show()
# im1.show()
# # im2=image("capture/capture.jpg.jpg")
# im1.compare(im)
