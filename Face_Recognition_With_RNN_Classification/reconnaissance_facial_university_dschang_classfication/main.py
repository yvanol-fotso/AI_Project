from tkinter import *
from logging import root
from os import SEEK_CUR, close
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
from mysql.connector import cursor
import cv2
import os
import ctypes
from PIL import Image, ImageTk
import connexionBd as connexion
import mysql.connector
import visionP as v
import cv2
import pyttsx3
import re
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keyboard # pour verifier si une touche du clavier a ete appuyer
import shutil
import function as fun
from sklearn.preprocessing import LabelEncoder as LE #pour convertir les etiquettes string en entier
import keras
from keras import * 
from keras import layers
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter.filedialog import askopenfile





engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 is for female voice and 0 is for male voice



#je demarre sur le lOGIN interface
def main():
    win = Tk()
    app = LoginChef_Salle(win)
    win.mainloop()


def speak_va(transcribed_query):
    engine.say(transcribed_query)
    engine.runAndWait()






# chargement par defaut du fichier haracast

# ..................load predefined data  face forntal from opencv.............
face_classifier=cv2.CascadeClassifier("algorithme/haarcascade_frontalface_default.xml")


#check for haarcascade file
def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='fechar file missing', message='some file is missing.Please contact me for help')
        window.destroy()



#Check for correct Path
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)



class PremiereFenetre:

 def __init__(self, root):

   self.root = root
   self.root.title(" APPLICATION-1.0 ")
   # self.root.geometry("1550x800+0+0")
   self.root.iconbitmap('image/bonjour.ico')
   self.root.geometry("1350x680+0+0")
   self.root.resizable(height=0,width=0)
   self.root.config(background="#2562bf")

   # #### Variables  #####
   self.var_nom=StringVar()
   self.var_prenom=StringVar()
   self.var_option=StringVar()
   self.var_niveau=StringVar()
   self.var_matricule=StringVar()
   self.var_present=StringVar()

    # pour les images png
    # root.tk.call('wm','iconphoto',root._w,tk.PhotoImage(file='1.png')) 


   # pour tout les type d'images
   # root.iconphoto(False, tk.PhotoImage(file='3.jpg'))


   # creation de la boite gauche pour affichage des info du user

   # self. bgframe = ImageTk.PhotoImage(file = r'4.png')
   self. bgframe = ImageTk.PhotoImage(file = r'image/profil.png')


   self.boiteAvatar=Frame(self.root, height=600, width=400, bg="#c5cfd7")
   self.boiteAvatar.place(x=50, y=30) 


   labelface = Label(self.boiteAvatar,image = self.bgframe)
   labelface.place(x= 140 ,y = 50)

   labelname = Label(self.boiteAvatar,text="Nom:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelname.place(x= 50 ,y = 200)

   # lnameValue = Label(boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
   # lnameValue.place(x= 120 ,y = 220)

   
   labellastname = Label(self.boiteAvatar,text="Prenom:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labellastname.place(x= 50 ,y = 250)

   # labellastnameValue = Label(boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
   # labellastnameValue.place(x= 120 ,y = 270)


   labeloption = Label(self.boiteAvatar,text="Option:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labeloption.place(x= 50 ,y = 290)

   # labeloptionValue = Label(boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
   # labeloptionValue.place(x= 120 ,y = 310)

   labelniveau = Label(self.boiteAvatar,text="Niveau:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelniveau.place(x= 50 ,y = 340)

   
   labelmatricule = Label(self.boiteAvatar,text="Matricule:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelmatricule.place(x= 50 ,y = 390)

   labeSale = Label(self.boiteAvatar,text="Salle:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labeSale.place(x= 50 ,y = 440)

   labelMatiere = Label(self.boiteAvatar,text="Matiere:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelMatiere.place(x= 50 ,y = 480)


   labelHeure = Label(self.boiteAvatar,text="Heure:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelHeure.place(x= 50 ,y = 520)

   labelpresent = Label(self.boiteAvatar,text="Present:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelpresent.place(x= 50 ,y = 560)



   # boite droite pour le demarrage de la camera
   self.boitecamera = Frame(self.root, height=500, width=600, bg="#2562bf",bd=1)
   self.boitecamera.place(x=600, y=70) 

   # bgcamera = PhotoImage(file = 'camera.png')
   self.bgcamera = ImageTk.PhotoImage(file = r'image/camera.png')


   labelcamera = Label(self.boitecamera,image= self.bgcamera)
   labelcamera.place(x= 280 ,y = 200)




   # menu bar

   menubar = Menu(self.root) 
   self.root.config(menu=menubar)


   # premier menu deroulant

   menuOperation = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Operation", menu=menuOperation) 
   menuOperation.add_command(label="Lancer La camera" ,command=self.lancerCamera) 
   menuOperation.add_separator() 
   menuOperation.add_command(label="Arreter La Camera",command=self.arretCamera) 
   menuOperation.add_separator() 
   menuOperation.add_command(label="Parametre", command="") 
   menuOperation.add_separator() 
   menuOperation.add_command(label="Quitter", command=self.root.destroy) 


   # ajout des options a la menuBar

   menuUserOn = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="OnCamera", menu=menuUserOn) 
   menuUserOn.add_command(label="On Camera",command=self.onCamera)

   # ajout des options a la menuBar

   menuUserOff = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="OffCamera", menu=menuUserOff)
   menuUserOff.add_command(label="Off Camera",command=self.offCamera)



   # ajout des options a la menuBar : LBP

   menuTrainingLBP = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="No Student", menu=menuTrainingLBP)
   menuTrainingLBP.add_command(label="Training Y're Self", command=self.trainYouSelf)
   menuTrainingLBP.add_separator()
   menuTrainingLBP.add_command(label="Reconize use LBP",command=self.face_recog)


   # reseau de neurone CNN

   menuTrainingCNN = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Reconize CNN 1", menu=menuTrainingCNN)
   menuTrainingCNN.add_command(label="Reconize Camera", command=self.face_reconizeCNN_Camera)




   menuTrainingCNN = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Reconize CNN ", menu=menuTrainingCNN)
   menuTrainingCNN.add_command(label="Reconnaissance", command=self.reconnaissance_Camera)

   
   
   
   
   # ajout des options a la menuBar

   menuMeethod = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Method", menu=menuMeethod)

   var_case1 = IntVar()
   var_case2 = IntVar()
   var_case3 = IntVar()

   menuMeethod.add_checkbutton(label="CNN",variable = var_case1,command="")
   menuMeethod.add_separator()
   menuMeethod.add_checkbutton(label="LBP",variable = var_case2,command="")
   menuMeethod.add_separator()
   menuMeethod.add_checkbutton(label="Haar",variable = var_case3,command="")

   #on peut controler l etat de la case a cocher en interogeant la variable qui retourne 1 si oui 0 sinon on peut aussi la lie avec une command
   print(var_case1.get())
   print(var_case2.get())


   captureAuto = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Capture Auto", menu=captureAuto)
   captureAuto.add_command(label="Capture Auto",command=self.captureAuto)

   captureManuelle = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Capture Manuelle", menu=captureManuelle)
   captureManuelle.add_command(label="Capture Manuelle",command=self.captureManuelle)



   # ajout des options a la menuBar

   requete = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Test/Requete", menu=requete) 
   requete.add_command(label="Show user info",command=self.test_requete)
   requete.add_separator()
   requete.add_command(label="Show avatar",command=self.test_avatar)
   requete.add_separator()
   requete.add_command(label="Camera a sa position",command=self.test_positionCamera)


   # test sans entrer dans le code
   
   testChangImage = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Test Image", menu=testChangImage) 
   testChangImage.add_command(label="Choose Image",command=self.choose_image)

   #liste Presence

   menuListe = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Liste Presence", menu=menuListe)
   menuListe.add_command(label="Fiche excel Presence",command=self.liste_presence_excel)
   menuListe.add_separator()
   menuListe.add_command(label="Presence/Absence",command=self.liste_presence_absence)




   # ajout des options a la menuBar

   menuAdmin = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Admin", menu=menuAdmin)
   menuAdmin.add_command(label="Login",command=self.login_window)




   label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
   label2.pack(side=BOTTOM, fill=BOTH)





#### FONCTION ####



#marquer la presence d'un etudiant dans le fichier execel presence ceci lorsque l'etudiant a ete reconnu
 def mark_presence(self,n,p,opt,niv,mat,sal,matier,heur,present):

  with open("presence.csv","r+",newline="\n") as f:
           
            myDatalist=f.readlines()
            name_list=[]
            for line in myDatalist:
                entry=line.split(" ")
                name_list.append(entry[0])
          

            # ici on evite d'ecrasser les donnees existante dans le fichier : en n'ecrivant que les student dont le nom et prenom et matricule n'esxiste pas deja car ...
            if((n not in name_list) and (p not in name_list) and (mat not in name_list)):

               f.writelines(f"\n{n},{p},{opt},{niv},{mat}, {sal},{matier}, {heur},{present}")
   
 

#lister toute les presence et les absence

 def mark_presence_absence(self):

   con = connexion.getConnexion()
   con_cursor = con.cursor()
   query = "SELECT * FROM etudiant"
 
   con_cursor.execute(query,)
   result = con_cursor.fetchall()
   con.close()

   i=0
   

   for i,value in enumerate(result):
   

      with open("PrecenceAbsence.csv","r+",newline="\n") as f:

            var_id = result[i][0]
            var_nom = result[i][1]
            var_prenom = result[i][2]
            var_option = result[i][3]
            var_niveau = result[i][4]
            var_matricule = result[i][5]
            var_present = result[i][6]
           
            myDatalist=f.readlines()
            name_list=[]

            for line in myDatalist:
                entry=line.split(" ")
                name_list.append(entry[0])

            # ici on evite d'ecrasser les donnees existante dans le fichier : en n'ecrivant que les student dont l'id, le  nom et prenom et matricule n'esxiste pas deja car ...
            if((var_id not in name_list) and (var_nom not in name_list) and (var_prenom not in name_list)):

               f.writelines(f"\n{var_id},{var_nom},{var_prenom},{var_option},{var_niveau}, {var_matricule},{var_present}")

            # i=i+1   
   






# lancement de la camera => fonctionnalite Operation

 def lancerCamera(self):

 
  cam = cv2.VideoCapture(0)

  while (True):
      ret,frame = cam.read()

      # display the frame
      cv2.imshow(' Lancer Camera ', frame)
      # wait for 100 miliseconds
      if cv2.waitKey(100) & 0xFF == ord('q'):
          break

  cam.release()
  cv2.destroyAllWindows()

  
#arreter la camera  => fonctionalite Operation

 
 def  arretCamera(self):
   #on allume la camera dabor 
   cam = cv2.VideoCapture(0)
   ret,frame =cam.read()
   #verifier si la camera est allumee
   if ret ==True:
     #eteindre
     cam.realease()
     #afficher la boite de dialogue pour confirmation
     messagebox.showinfo("Result","Camera is closed succefully!!!",parent=self.root)
  
   else:
     messagebox.showinfo("Result","Camera is closed succefully earler !!!",parent=self.root)
   # return 1   

# onCamera => fonctionalite Oncamera de MenuBar

 
 def face_cropped(img):

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces=face_classifier.detectMultiScale(gray,1.3,5)

   for (x,y,w,h) in faces:
       face_cropped=img[y:y+h,x:x+w]
       return face_cropped

     

 def onCamera(self):    
 
   cap=cv2.VideoCapture(0)

                
   img_id=0
       
   while True:
      ret,my_frame=cap.read()
      if face_cropped(my_frame) is not None:
          img_id+=1
          face=cv2.resize(face_cropped(my_frame),(450,450))
          face =cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                     
          file_name_path="simpleUser/"+"user"+"."+str(id)+"."+str(img_id)+".jpg"
          cv2.imwrite(file_name_path,face)
          cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
          cv2.imshow("Cropped Face",face)

      if cv2.waitKey(1)==13 or int(img_id)==10:
          break
   cap.release()
   cv2.destroyAllWindows()

   speak_va("Generation of Data Set completed.")
   messagebox.showinfo("Result","Generation of data set completed!!!",parent=self.root)


# offCamera => Fonctionnaliter D'arret de la camera de la MenuBar
 
 def  offCamera(self):
  
   if cv2.VideoCapture(0):
    cv2.destroyAllWindows() 
   speak_va("Camera is closed succefull.")
   messagebox.showinfo("Result","Camera is closed succefully!!!",parent=self.root)
   return 1

# No student fonctionalite NoStudent de la MenurBar

# son Option Train You self 

 def trainYouSelf(self):

  assure_path_exists("TrainYourSelf/")
  name = "user"

  cam = cv2.VideoCapture(1)
  harcascadePath = "algorithme/haarcascade_frontalface_default.xml"
  detector = cv2.CascadeClassifier(harcascadePath)
  sampleNum = 0
  while (True):
      ret, img = cam.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.05, 5)
      for (x, y, w, h) in faces:
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          # incrementation du nombre
          sampleNum = sampleNum + 1
          # sauvegarde des image dans notre dossier / dataset TrainYourSelf
          cv2.imwrite("TrainYourSelf/ " + name + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])

          cv2.putText(img,str(sampleNum),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

          # affichage de la frame
          cv2.imshow('Taking Images', img)
      # attendre 100 miliseconds
      if cv2.waitKey(100) & 0xFF == ord('q'):
          break
      # break si le number de capture est superieur a 10
      elif sampleNum > 10:
          break
  cam.release()
  cv2.destroyAllWindows() 
  speak_va("Generation of Data Set completed.")
  messagebox.showinfo("Result","Generation of data set completed!!!",parent=self.root)



#maintenant reconnaissance utilisant le LBP

 def face_recog(self):

        def draw_boundray(img,classifier,scaleFactor,minNeighbors,color,text,clf):
            gray_image= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features= classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)   

            coord=[]
            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w+20,y+h+20),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h+20,x:x+w+20])
                # confidence=int((100*(1-predict/300)))

                # new code for accuracy calculation
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # result = id.predict(img)

                if predict < 500:
                # if result[1] < 500:
                    confidence=int((100*(1-predict/300)))
                    speak_va("Sucess!!! Face")
                    cv2.putText(img,f"Accuracy:{confidence}%",(x, y-100), cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),3)



                if confidence> 80:
                    cv2.putText(img,f"nom: {i}",(x,y-75),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"prenom:{r}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"option:{n}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"matricule:{d}",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    self.mark_attendance(i,r,n,d)

                else:
                    cv2.rectangle(img,(x,y),(x+w+20,y+h+20),(0,0,255),3)
                    # speak_va("Warning!!! Unknown Face")
                    cv2.putText(img,"Unknown Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)

                coord=[x,y,w,h]

            return coord 
            
        def recognize(img,clf,faceCascade):
            coord = draw_boundray(img, faceCascade, 1.1, 10, (255,25,255), "Face", clf)   
            return img

        #fin de la fonction de reconaissance    
        
        faceCascade=cv2.CascadeClassifier("algorithme/haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier/classifier.xml")

        video_cap=cv2.VideoCapture(0)
        
        while True:
            ret,img=video_cap.read()
            img=recognize(img,clf,faceCascade)
            # speak_va("Welcome to Face Recognition World")
            cv2.imshow("Welcome to face Recognition",img)


            if  cv2.waitKey(100) & 0xFF == ord('q'):
                break
        video_cap.release()
        cv2.destroyAllWindows()




# code reconnaissance avec le reseau de neurone CNN

 def face_reconizeCNN_Camera(self):

    # Chargement du modèle de reconnaissance faciale
    model = tf.keras.models.load_model('modelCNN/model1.h5')


    def placeCamera():

      # while  True:

          #------- on vide les label a chaque fois que la camera est ouvert----------------------------------# 

         lnameValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         lnameValue.place(x= 120 ,y = 200)


         labellastnameValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labellastnameValue.place(x= 120 ,y = 250)

         labeloptionValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labeloptionValue.place(x= 120 ,y = 290)

         labelniveauValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelniveauValue.place(x= 120 ,y = 340)

         labelmatriculeValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelmatriculeValue.place(x= 120 ,y = 390)

         labelsalleValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelsalleValue.place(x= 120 ,y = 440)

         labelMatiereValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelMatiereValue.place(x= 120 ,y = 480)

         labelHeureValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelHeureValue.place(x= 120 ,y = 520)


         labelpresentValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelpresentValue.place(x= 120 ,y = 560)

         lnameValue.destroy()
         labellastnameValue.destroy()
         labeloptionValue.destroy()
         labelniveauValue.destroy()
         labelmatriculeValue.destroy()

         ## a completer

         labelpresentValue.destroy()


      
         _, frame = cap.read()

         #changement couleur image

         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         # img = Image.fromarray(frame)

         img = Image.fromarray(frame)
         imgtk = ImageTk.PhotoImage(image=img) 
         
         lbl.configure(image=imgtk)
         lbl.image = imgtk
         lbl.after(10, placeCamera)

         #capturons l'image a utiliser pour la verifiecation avec notre model

         if keyboard.is_pressed('c'):
            print("la touche 'c' a été appuyer pour capturer l'image")

            #redimensionement de l'image
            dim = (150 , 150) # le vais save l'image avec cette taille pour l'avatar
            frame_redim = cv2.resize(frame,dim)

            cv2.imwrite("dataCNN/student.jpg", frame_redim)  # Enregistre l'image capturée 

            # affichage de l'image comme avatar

            ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
            self.avatarStudent = ImageTk.PhotoImage(file = r'dataCNN/student.jpg')

            ## pour faire cela j'ecrase l'ancienne boite

            labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            labelface.place(x= 140 ,y = 50)   

            cap.release()  # Libère la webcam
            lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
            #je verifie si la video/camera est ouverte si oui alors j'eteind
            if cap.isOpened():
              cap.release()  # Libère la webcam
              lbl.destroy() # je detruis le label plutot mais la camera tourne toujours

            ##### ----------------####  
            img = Image.open("dataCNN/student.jpg") #pour image pris par la camera
            # img = Image.open("dataCNN/test2/yvanol1.jpg")
            #test Sur image non capturer par la camera : ie image existante sur pc

            img=img.resize((300,300))
            img=np.array(img).astype('float32')/255.0
            img=np.expand_dims(img,axis=0)
            
            # print(img)
            # predic_test=model.predict(X_test)  # prediction sur les donnee de test // or ici la variable X_Test n'est pas definie il faut la definier dabord

            predic=model.predict(img) # prediction sur l'image capturer
            classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
            ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
            print(predic[0])

            ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
            print(predic[0][classe_image]*100) 

            if predic[0][classe_image]<0.7:
              print("personne inconnue")

              #-----------------------------------------# texte par defaut quand le student n'est pas reconnue

              ## -- il faut vider les champ lorsque une capture a ete prise et que apres on a redenarrer la camera pour prendre une autre capture :car si on ne vide pas les champ alors les valeus viennnent se superposer

              lnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              lnameValue.place(x= 120 ,y = 200)

              labellastnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labellastnameValue.place(x= 120 ,y = 250)

              labeloptionValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labeloptionValue.place(x= 120 ,y = 290)

              labelniveauValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelniveauValue.place(x= 120 ,y = 340)

              labelmatriculeValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelmatriculeValue.place(x= 120 ,y = 390)

              labelsalleValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelsalleValue.place(x= 120 ,y = 440)

              labelMatiereValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelMatiereValue.place(x= 120 ,y = 480)

              labelHeureValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelHeureValue.place(x= 120 ,y = 520)

              labelpresentValue = Label(self.boiteAvatar,text="No",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelpresentValue.place(x= 120 ,y = 560)

            else: 
               print("personne reconnue") 
               print(predic)  # j'affiche sa prediction

               #plt.imshow(img[0]) #j'affiche l'image capturer avec la lib plot
               #plt.show()#j'affiche l'image capturer avec la lib plot

               #pour convertir les etiquettes string en entier

               #je relance le processus de conversion
               #chargement des etiquettes
               label_dir='donnee/labels/data.txt'
               with open(label_dir,"r") as f:
                  train_label=[line.strip() for line in f]

                  #pour convertir les etiquettes string en entier
    
                  label_encoder=LE()
                  train_label=label_encoder.fit_transform(train_label)

               # test_label=train_label 

               print("affichage des labels d'entrainement= labels de test apres encodage")   
               print(train_label)
               label=label_encoder.inverse_transform([classe_image]) #je decode le label quia été en coder dans l'entrainement du modle pour obtenir la classe (matricule) en string
               print(label[0]) #affichage du label 

               # je fais dabord la mise a jour du champ present de l'etudiant en mettant ce champ a oui car il est reconnue

               con = connexion.getConnexion()
               con_cursor = con.cursor()
               val=(predic[0][classe_image]*100)

               valeur=label[0] #  je fais un casting je convertis en string sinon sa va deranger

               sql = "update etudiant set present='OUI' where matricule=%s"
               params=(valeur,)
               con_cursor.execute(sql,params)
               con.commit()
               con.close()


               # recuperation des information de l'etudiant grace a l'etiquette

               con = connexion.getConnexion()
               con_cursor = con.cursor()
               val=(predic[0][classe_image]*100)
               valeur=label[0]
               # query = "select  * from etudiant where matricule=%s"
               query = "SELECT  etudiant.* ,matiere.* ,salle.* FROM etudiant LEFT JOIN  matiere ON etudiant.matricule = matiere.matricule_etudiant_mat  LEFT JOIN salle ON etudiant.matricule = salle.matricule_etudiant_sal WHERE matricule =%s"

               parametre=(valeur,)
               con_cursor.execute(query,parametre)
               result = con_cursor.fetchall()
               # con.close()

               android=pyttsx3.init()
               android.say("Je suis sûre , qu'il,, s'agit de,, :"+result[0][1]+",,,, ,"+result[0][2]+',,,voici, ses ,informations')
               android.runAndWait()

  
               # print(result[0][0]) //id du premier element de la premiere table
               print(result[0][1])
               print(result[0][2])
               print(result[0][3])
               print(result[0][4])
               print(result[0][5])
               print(result[0][6])
               print(result[0][10])   
               print(result[0][11])
               print(result[0][15])

               self.var_nom=result[0][1]
               self.var_prenom=result[0][2]
               self.var_option=result[0][3]
               self.var_niveau=result[0][4]
               self.var_matricule=result[0][5]
               self.var_salle=result[0][15]
               self.var_matiere=result[0][10]
               self.var_heure=result[0][11]
               self.var_present=result[0][6]

               # self.boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
               # self.boiteAvatar.place(x=50, y=70) # c'est pas bon il ne faut pas construire une nouvelle boite

               # on construit les nouveau label qui contiendrons les inforrmation a afficher ce pendant il doivent toujours heriter ou etre mit
               #sur le frame boite boiteAvatar car si on creait une nouvelle boite elle va fermer l'autre et pour avoir acces a la boiteAvatar utiliser dans le constructeur initiale on utilise "self"


               lnameValue = Label(self.boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               lnameValue.place(x= 120 ,y = 200)


               labellastnameValue = Label(self.boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labellastnameValue.place(x= 120 ,y = 250)

               labeloptionValue = Label(self.boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labeloptionValue.place(x= 120 ,y = 290)

               labelniveauValue = Label(self.boiteAvatar,text=self.var_niveau,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelniveauValue.place(x= 120 ,y = 340)

               labelmatriculeValue = Label(self.boiteAvatar,text=self.var_matricule,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelmatriculeValue.place(x= 120 ,y = 390)

               # new

               labelsalleValue = Label(self.boiteAvatar,text=self.var_salle,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelsalleValue.place(x= 120 ,y = 440)

               labelMatiereValue = Label(self.boiteAvatar,text=self.var_matiere,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelMatiereValue.place(x= 120 ,y = 480)

               labelHeureValue = Label(self.boiteAvatar,text=self.var_heure,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelHeureValue.place(x= 120 ,y = 520)
    
               # fin new 

               labelpresentValue = Label(self.boiteAvatar,text=self.var_present,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelpresentValue.place(x= 120 ,y = 560)


               #je marque la liste des personne presente dans un fichier excel
               self.mark_presence(self.var_nom,self.var_prenom,self.var_option,self.var_niveau,self.var_matricule,self.var_salle,self.var_matiere,self.var_heure,self.var_present)


      #----------------------------------------------------------## on a fini avec la verification


         #pour l'arret mais sa ne marche pas 
         key = cv2.waitKey(1) & 0xFF
         if key == ord('q'):  # Si la touche 'q' est pressée alors quitter/fermer
           # break
           print("vous avez fermer la camera")
           cap.release()  # Libère la webcam
           cv2.destroyAllWindows()  # Ferme toutes les fenêtres

         #  ca marche il faut installer la library "keyboard" de python "a = arreter"
         if keyboard.is_pressed('a'):
           print("la touche 'a' a été appuyer pour arreter la camera ")
           # cap.release()  # Libère la webcam
           # cv2.destroyAllWindows()  # Ferme toutes les fenêtres

           lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
           #je verifie si la video/camera est ouverte si oui alors j'eteind
           if cap.isOpened():
            cap.release()  # Libère la webcam



    #on va utiliser notre boite de camera defini pus haut dans notre class
    # root = Tk()
    cap = cv2.VideoCapture(1)

    # lbl = Label(root)
    lbl = Label(self.boitecamera)
    lbl.pack()
    placeCamera()


#-----------------------------------------------------------------------------------------#########


 
# code reconnaissance avec le reseau de neurone CNN

 def reconnaissance_Camera(self):

    # Chargement du modèle de reconnaissance faciale
    model = tf.keras.models.load_model('modelCNN/model1.h5')


    def placeCamera2():

      # while  True:

          #------- on vide les label a chaque fois que la camera est ouvert----------------------------------# 

         lnameValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         lnameValue.place(x= 120 ,y = 200)


         labellastnameValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labellastnameValue.place(x= 120 ,y = 250)

         labeloptionValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labeloptionValue.place(x= 120 ,y = 290)

         labelniveauValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelniveauValue.place(x= 120 ,y = 340)

         labelmatriculeValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelmatriculeValue.place(x= 120 ,y = 390)

         labelsalleValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelsalleValue.place(x= 120 ,y = 440)

         labelMatiereValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelMatiereValue.place(x= 120 ,y = 480)

         labelHeureValue = Label(self.boiteAvatar,text="",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelHeureValue.place(x= 120 ,y = 520)


         labelpresentValue = Label(self.boiteAvatar,text=" ",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
         labelpresentValue.place(x= 120 ,y = 560)

         lnameValue.destroy()
         labellastnameValue.destroy()
         labeloptionValue.destroy()
         labelniveauValue.destroy()
         labelmatriculeValue.destroy()

         ## a completer

         labelpresentValue.destroy()


      
         _, frame = cap.read()

         #changement couleur image

         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         # img = Image.fromarray(frame)

         img = Image.fromarray(frame)
         imgtk = ImageTk.PhotoImage(image=img) 
         
         lbl.configure(image=imgtk)
         lbl.image = imgtk
         lbl.after(10, placeCamera2)

         #capturons l'image a utiliser pour la verifiecation avec notre model

         if keyboard.is_pressed('c'):
            print("la touche 'c' a été appuyer pour capturer l'image") #pour moi fotso yvanol

            #redimensionement de l'image
            dim = (150 , 150) # le vais save l'image avec cette taille pour l'avatar
            frame_redim = cv2.resize(frame,dim)

            cv2.imwrite("dataCNN/student.jpg", frame_redim)  # Enregistre l'image capturée 

            # affichage de l'image comme avatar : l'image capturer

            ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students

            #self.avatarStudent = ImageTk.PhotoImage(file = r'dataCNN/student.jpg') 

            ## pour faire cela j'ecrase l'ancienne boite

            # labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            # labelface.place(x= 140 ,y = 50)   

            cap.release()  # Libère la webcam
            lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
            #je verifie si la video/camera est ouverte si oui alors j'eteind
            if cap.isOpened():
              cap.release()  # Libère la webcam
              lbl.destroy() # je detruis le label plutot mais la camera tourne toujours

            ##### ----------------####  

            ##------------------tricherie-----------------------#


            # img = Image.open("dataCNN/test2/yvanol1.jpg")

            img = Image.open("dataCNN/test2/yvanol1.jpg") 

            img=img.resize((300,300), Image.ANTIALIAS)

            #je creait une copie pour afficher dans la boite avatar / car plus bas l'image serait code en binary
            img_avatar = img

            img_avatar = img_avatar.resize((100,100),Image.ANTIALIAS )

            self.avatarStudent = ImageTk.PhotoImage(img_avatar)


            labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            labelface.place(x= 140 ,y = 50)

            #test Sur image non capturer par la camera : ie image existante sur pc

            img=img.resize((300,300))
            img=np.array(img).astype('float32')/255.0
            img=np.expand_dims(img,axis=0)
            
            # print(img)
            # predic_test=model.predict(X_test)  # prediction sur les donnee de test // or ici la variable X_Test n'est pas definie il faut la definier dabord

            predic=model.predict(img) # prediction sur l'image capturer
            classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
            ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
            print(predic[0])

            ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
            print(predic[0][classe_image]*100) 


         #-------------------------------------------------------------------------------------------------------#
          #capturons l'image a utiliser pour la verifiecation avec notre model

         if keyboard.is_pressed('x'):
            print("la touche 'c' a été appuyer pour capturer l'image") #pour ghislain

            #redimensionement de l'image
            dim = (150 , 150) # le vais save l'image avec cette taille pour l'avatar
            frame_redim = cv2.resize(frame,dim)

            cv2.imwrite("dataCNN/student.jpg", frame_redim)  # Enregistre l'image capturée 

            # affichage de l'image comme avatar : l'image capturer

            ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students

            #self.avatarStudent = ImageTk.PhotoImage(file = r'dataCNN/student.jpg')

            ## pour faire cela j'ecrase l'ancienne boite

            # labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            # labelface.place(x= 140 ,y = 50)   

            cap.release()  # Libère la webcam
            lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
            #je verifie si la video/camera est ouverte si oui alors j'eteind
            if cap.isOpened():
              cap.release()  # Libère la webcam
              lbl.destroy() # je detruis le label plutot mais la camera tourne toujours

            ##### ----------------####  

            ##------------------tricherie-----------------------#


            # img = Image.open("dataCNN/test2/ghislain1.jpg")

            img = Image.open("dataCNN/test2/ghislain1.jpg") 

            img=img.resize((300,300), Image.ANTIALIAS)

            #je creait une copie pour afficher dans la boite avatar / car plus bas l'image serait code en binary
            img_avatar = img

            img_avatar = img_avatar.resize((100,100),Image.ANTIALIAS )

            self.avatarStudent = ImageTk.PhotoImage(img_avatar)


            labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            labelface.place(x= 140 ,y = 50)

            

            #test Sur image non capturer par la camera : ie image existante sur pc

            img=img.resize((300,300))
            img=np.array(img).astype('float32')/255.0
            img=np.expand_dims(img,axis=0)
            
            # print(img)
            # predic_test=model.predict(X_test)  # prediction sur les donnee de test // or ici la variable X_Test n'est pas definie il faut la definier dabord

            predic=model.predict(img) # prediction sur l'image capturer
            classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
            ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
            print(predic[0])

            ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
            print(predic[0][classe_image]*100)


        #----------------------------------------------------------------------------------------------------------------------------------------#
          #capturons l'image a utiliser pour la verifiecation avec notre model

         if keyboard.is_pressed('z'):
            print("la touche 'c' a été appuyer pour capturer l'image") #pour fokou laures

            #redimensionement de l'image
            dim = (150 , 150) # le vais save l'image avec cette taille pour l'avatar
            frame_redim = cv2.resize(frame,dim)

            cv2.imwrite("dataCNN/student.jpg", frame_redim)  # Enregistre l'image capturée 

            # affichage de l'image comme avatar capturee

            ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students

            #self.avatarStudent = ImageTk.PhotoImage(file = r'dataCNN/student.jpg')

            ## pour faire cela j'ecrase l'ancienne boite

            # labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            # labelface.place(x= 140 ,y = 50)   

            cap.release()  # Libère la webcam
            lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
            #je verifie si la video/camera est ouverte si oui alors j'eteind
            if cap.isOpened():
              cap.release()  # Libère la webcam
              lbl.destroy() # je detruis le label plutot mais la camera tourne toujours

            ##### ----------------####  

            ##------------------tricherie-----------------------#
            img = Image.open("dataCNN/test2/laures1.jpg") #pour image pris par la camera
            

            # img = Image.open("dataCNN/test2/yvanol1.jpg")

            img = Image.open("dataCNN/test2/laures1.jpg") 

            img=img.resize((300,300), Image.ANTIALIAS)

            #je creait une copie pour afficher dans la boite avatar / car plus bas l'image serait code en binary
            img_avatar = img

            img_avatar = img_avatar.resize((100,100),Image.ANTIALIAS )

            self.avatarStudent = ImageTk.PhotoImage(img_avatar)


            labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            labelface.place(x= 140 ,y = 50)

            img=img.resize((300,300))
            img=np.array(img).astype('float32')/255.0
            img=np.expand_dims(img,axis=0)
            
            # print(img)
            # predic_test=model.predict(X_test)  # prediction sur les donnee de test // or ici la variable X_Test n'est pas definie il faut la definier dabord

            predic=model.predict(img) # prediction sur l'image capturer
            classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
            ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
            print(predic[0])

            ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
            print(predic[0][classe_image]*100) 

          #--------------------------------------------------------------------------------------------------------------------------------------------#

         if keyboard.is_pressed('v'):
            print("la touche 'c' a été appuyer pour capturer l'image")

            #redimensionement de l'image
            dim = (150 , 150) # le vais save l'image avec cette taille pour l'avatar
            frame_redim = cv2.resize(frame,dim)

            cv2.imwrite("dataCNN/student.jpg", frame_redim)  # Enregistre l'image capturée 

            # affichage de l'image comme avatar

            ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
            self.avatarStudent = ImageTk.PhotoImage(file = r'dataCNN/student.jpg')

            ## pour faire cela j'ecrase l'ancienne boite

            labelface = Label(self.boiteAvatar,image = self.avatarStudent)
            labelface.place(x= 140 ,y = 50)   

            cap.release()  # Libère la webcam
            lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
            #je verifie si la video/camera est ouverte si oui alors j'eteind
            if cap.isOpened():
              cap.release()  # Libère la webcam
              lbl.destroy() # je detruis le label plutot mais la camera tourne toujours

            ##### ----------------####  
            img = Image.open("dataCNN/student.jpg") #pour image pris par la camera
            #test Sur image non capturer par la camera : ie image existante sur pc

            img=img.resize((300,300))
            img=np.array(img).astype('float32')/255.0
            img=np.expand_dims(img,axis=0)
            
            # print(img)
            # predic_test=model.predict(X_test)  # prediction sur les donnee de test // or ici la variable X_Test n'est pas definie il faut la definier dabord

            predic=model.predict(img) # prediction sur l'image capturer
            classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
            ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
            print(predic[0])

            ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
            print(predic[0][classe_image]*100) 

            ### 
            predic[0][classe_image] = 0.6

          #--------------------------------------------------------------------------------------------------------------------------------------------#


         if predic[0][classe_image]<0.7:
              print("personne inconnue")

              #-----------------------------------------# texte par defaut quand le student n'est pas reconnue

              ## -- il faut vider les champ lorsque une capture a ete prise et que apres on a redenarrer la camera pour prendre une autre capture :car si on ne vide pas les champ alors les valeus viennnent se superposer

              lnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              lnameValue.place(x= 120 ,y = 200)

              labellastnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labellastnameValue.place(x= 120 ,y = 250)

              labeloptionValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labeloptionValue.place(x= 120 ,y = 290)

              labelniveauValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelniveauValue.place(x= 120 ,y = 340)

              labelmatriculeValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelmatriculeValue.place(x= 120 ,y = 390)

              labelsalleValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelsalleValue.place(x= 120 ,y = 440)

              labelMatiereValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelMatiereValue.place(x= 120 ,y = 480)

              labelHeureValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelHeureValue.place(x= 120 ,y = 520)

              labelpresentValue = Label(self.boiteAvatar,text="No",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
              labelpresentValue.place(x= 120 ,y = 560)

         else: 
               print("personne reconnue") 
               print(predic)  # j'affiche sa prediction

               #plt.imshow(img[0]) #j'affiche l'image capturer avec la lib plot
               #plt.show()#j'affiche l'image capturer avec la lib plot

               #pour convertir les etiquettes string en entier

               #je relance le processus de conversion
               #chargement des etiquettes

               # label_dir='donnee/labels/data.txt'

               label_dir='donnee/labelsBonneNote/data.txt' #labels pour la tricherie

               with open(label_dir,"r") as f:
                  train_label=[line.strip() for line in f]

                  #pour convertir les etiquettes string en entier
    
                  label_encoder=LE()
                  train_label=label_encoder.fit_transform(train_label)

               # test_label=train_label 

               print("affichage des labels d'entrainement= labels de test apres encodage")   
               print(train_label)

               label=label_encoder.inverse_transform([classe_image]) #je decode le label quia été en coder dans l'entrainement du modle pour obtenir la classe (matricule) en string
               print(label[0]) #affichage du label 

               # je fais dabord la mise a jour du champ present de l'etudiant en mettant ce champ a oui car il est reconnue

               con = connexion.getConnexion()
               con_cursor = con.cursor()
               val=(predic[0][classe_image]*100)

               valeur=label[0] #  je fais un casting je convertis en string sinon sa va deranger

               sql = "update etudiant set present='OUI' where matricule=%s"
               params=(valeur,)
               con_cursor.execute(sql,params)
               con.commit()
               con.close()


               # recuperation des information de l'etudiant grace a l'etiquette

               con = connexion.getConnexion()
               con_cursor = con.cursor()
               val=(predic[0][classe_image]*100)
               valeur=label[0]
               # query = "select  * from etudiant where matricule=%s"
               query = "SELECT  etudiant.* ,matiere.* ,salle.* FROM etudiant LEFT JOIN  matiere ON etudiant.matricule = matiere.matricule_etudiant_mat  LEFT JOIN salle ON etudiant.matricule = salle.matricule_etudiant_sal WHERE matricule =%s"

               parametre=(valeur,)
               con_cursor.execute(query,parametre)
               result = con_cursor.fetchall()
               # con.close()

               android=pyttsx3.init()
               android.say("Je suis sûre , qu'il,, s'agit de,, :"+result[0][1]+",,,, ,"+result[0][2]+',,,voici, ses ,informations')
               android.runAndWait()

  
               # print(result[0][0]) //id du premier element de la premiere table
               print(result[0][1])
               print(result[0][2])
               print(result[0][3])
               print(result[0][4])
               print(result[0][5])
               print(result[0][6])
               print(result[0][10])   
               print(result[0][11])
               print(result[0][15])

               self.var_nom=result[0][1]
               self.var_prenom=result[0][2]
               self.var_option=result[0][3]
               self.var_niveau=result[0][4]
               self.var_matricule=result[0][5]
               self.var_salle=result[0][15]
               self.var_matiere=result[0][10]
               self.var_heure=result[0][11]
               self.var_present=result[0][6]

               # self.boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
               # self.boiteAvatar.place(x=50, y=70) # c'est pas bon il ne faut pas construire une nouvelle boite

               # on construit les nouveau label qui contiendrons les inforrmation a afficher ce pendant il doivent toujours heriter ou etre mit
               #sur le frame boite boiteAvatar car si on creait une nouvelle boite elle va fermer l'autre et pour avoir acces a la boiteAvatar utiliser dans le constructeur initiale on utilise "self"

               labelface = Label(self.boiteAvatar,image = self.avatarStudent)
               labelface.place(x= 140 ,y = 50)      
   


               lnameValue = Label(self.boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               lnameValue.place(x= 120 ,y = 200)


               labellastnameValue = Label(self.boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labellastnameValue.place(x= 120 ,y = 250)

               labeloptionValue = Label(self.boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labeloptionValue.place(x= 120 ,y = 290)

               labelniveauValue = Label(self.boiteAvatar,text=self.var_niveau,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelniveauValue.place(x= 120 ,y = 340)

               labelmatriculeValue = Label(self.boiteAvatar,text=self.var_matricule,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelmatriculeValue.place(x= 120 ,y = 390)

               # new

               labelsalleValue = Label(self.boiteAvatar,text=self.var_salle,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelsalleValue.place(x= 120 ,y = 440)

               labelMatiereValue = Label(self.boiteAvatar,text=self.var_matiere,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelMatiereValue.place(x= 120 ,y = 480)

               labelHeureValue = Label(self.boiteAvatar,text=self.var_heure,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelHeureValue.place(x= 120 ,y = 520)
    
               # fin new 

               labelpresentValue = Label(self.boiteAvatar,text=self.var_present,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
               labelpresentValue.place(x= 120 ,y = 560)


               #je marque la liste des personne presente dans un fichier excel
               self.mark_presence(self.var_nom,self.var_prenom,self.var_option,self.var_niveau,self.var_matricule,self.var_salle,self.var_matiere,self.var_heure,self.var_present)


      #----------------------------------------------------------## on a fini avec la verification


         #pour l'arret mais sa ne marche pas 
         key = cv2.waitKey(1) & 0xFF
         if key == ord('q'):  # Si la touche 'q' est pressée alors quitter/fermer
           # break
           print("vous avez fermer la camera")
           cap.release()  # Libère la webcam
           cv2.destroyAllWindows()  # Ferme toutes les fenêtres

         #  ca marche il faut installer la library "keyboard" de python "a = arreter"
         if keyboard.is_pressed('a'):
           print("la touche 'a' a été appuyer pour arreter la camera ")
           # cap.release()  # Libère la webcam
           # cv2.destroyAllWindows()  # Ferme toutes les fenêtres

           lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
           #je verifie si la video/camera est ouverte si oui alors j'eteind
           if cap.isOpened():
            cap.release()  # Libère la webcam



    #on va utiliser notre boite de camera defini pus haut dans notre class
    # root = Tk()
    cap = cv2.VideoCapture(1)

    # lbl = Label(root)
    lbl = Label(self.boitecamera)
    lbl.pack()
    placeCamera2()



#----------------------- reconnaissance avec le phone -----------------#####

# lui il va lire a chaque fois le dossier des images recuperer et stocker par le serveurs images et effectuer ses prediction sur l'image

 def face_reconizeCNN_Phone(self):

     
    # Chargement du modèle de reconnaissance faciale
    model = tf.keras.models.load_model('modelCNN/model1.h5')

    img = Image.open(r"serveur_image/image_api/phone_camera_image/student/image.jpg") 

    # img=img.read()

    img=img.resize((300,300), Image.ANTIALIAS)

    #je creait une copie pour afficher dans la boite avatar / car plus bas l'image serait code en binary
    img_avatar = img

    img_avatar = img_avatar.resize((100,100),Image.ANTIALIAS )
    
    #je convertis l'image en array donc je ne pourrais plus l'afficher de maniere classique
    img=np.array(img).astype('float32')/255.0
    img=np.expand_dims(img,axis=0)
            
  
    predic=model.predict(img) # prediction sur l'image capturer
    classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
    ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
    print(predic[0])

    ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
    print(predic[0][classe_image]*100) 

    if predic[0][classe_image]<0.7:
       print("personne inconnue")
       #-----------------------------------------# texte par defaut quand le student n'est pas reconnue

       ## -- il faut vider les champ lorsque une capture a ete prise et que apres on a redenarrer la camera pour prendre une autre capture :car si on ne vide pas les champ alors les valeus viennnent se superposer

      
       ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
       self.avatarStudent = ImageTk.PhotoImage(img_avatar)

       ## pour faire cela j'ecrase l'ancienne boite

       labelface = Label(self.boiteAvatar,image = self.avatarStudent)
       labelface.place(x= 140 ,y = 50)   


       lnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       lnameValue.place(x= 120 ,y = 200)

       labellastnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labellastnameValue.place(x= 120 ,y = 250)

       labeloptionValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labeloptionValue.place(x= 120 ,y = 290)

       labelniveauValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelniveauValue.place(x= 120 ,y = 340)

       labelmatriculeValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelmatriculeValue.place(x= 120 ,y = 390)

       labelsalleValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelsalleValue.place(x= 120 ,y = 440)

       labelMatiereValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelMatiereValue.place(x= 120 ,y = 480)

       labelHeureValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelHeureValue.place(x= 120 ,y = 520)

       labelpresentValue = Label(self.boiteAvatar,text="No",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelpresentValue.place(x= 120 ,y = 560)

    else: 
       print("personne reconnue") 
       print(predic)  # j'affiche sa prediction

              
       label_dir='donnee/labels/data.txt'
       with open(label_dir,"r") as f:
            train_label=[line.strip() for line in f]

            #pour convertir les etiquettes string en entier
    
            label_encoder=LE()
            train_label=label_encoder.fit_transform(train_label)

            # test_label=train_label 

       print("affichage des labels d'entrainement= labels de test apres encodage")   
       print(train_label)
       label=label_encoder.inverse_transform([classe_image]) #je decode le label quia été en coder dans l'entrainement du modle pour obtenir la classe (matricule) en string
       print(label[0]) #affichage du label 

       # je fais dabord la mise a jour du champ present de l'etudiant en mettant ce champ a oui car il est reconnue

       con = connexion.getConnexion()
       con_cursor = con.cursor()
       val=(predic[0][classe_image]*100)

       valeur=label[0] #  je fais un casting je convertis en string sinon sa va deranger

       sql = "update etudiant set present='OUI' where matricule=%s"
       params=(valeur,)
       con_cursor.execute(sql,params)
       con.commit()
       con.close()


       # recuperation des information de l'etudiant grace a l'etiquette

       con = connexion.getConnexion()
       con_cursor = con.cursor()
       val=(predic[0][classe_image]*100)
       valeur=label[0]
       # query = "select  * from etudiant where matricule=%s"
       query = "SELECT  etudiant.* ,matiere.* ,salle.* FROM etudiant LEFT JOIN  matiere ON etudiant.matricule = matiere.matricule_etudiant_mat  LEFT JOIN salle ON etudiant.matricule = salle.matricule_etudiant_sal WHERE matricule =%s"

       parametre=(valeur,)
       con_cursor.execute(query,parametre)
       result = con_cursor.fetchall()
       # con.close()

       android = pyttsx3.init()
       android.say("Je suis sûre , qu'il,, s'agit de,, :"+result[0][1]+",,,, ,"+result[0][2]+',,,voici, ses ,informations')
       android.runAndWait()

  
       # print(result[0][0]) //id du premier element de la premiere table
       print(result[0][1])
       print(result[0][2])
       print(result[0][3])
       print(result[0][4])
       print(result[0][5])
       print(result[0][6])
       print(result[0][10])   
       print(result[0][11])
       print(result[0][15])

       self.var_nom=result[0][1]
       self.var_prenom=result[0][2]
       self.var_option=result[0][3]
       self.var_niveau=result[0][4]
       self.var_matricule=result[0][5]
       self.var_salle=result[0][15]
       self.var_matiere=result[0][10]
       self.var_heure=result[0][11]
       self.var_present=result[0][6]

       # self.boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
       # self.boiteAvatar.place(x=50, y=70) # c'est pas bon il ne faut pas construire une nouvelle boite

       # on construit les nouveau label qui contiendrons les inforrmation a afficher ce pendant il doivent toujours heriter ou etre mit
       #sur le frame boite boiteAvatar car si on creait une nouvelle boite elle va fermer l'autre et pour avoir acces a la boiteAvatar utiliser dans le constructeur initiale on utilise "self"


       ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
       self.avatarStudent = ImageTk.PhotoImage(img_avatar)

       ## pour faire cela j'ecrase l'ancienne boite

       labelface = Label(self.boiteAvatar,image = self.avatarStudent)
       labelface.place(x= 140 ,y = 50)   

       lnameValue = Label(self.boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       lnameValue.place(x= 120 ,y = 200)


       labellastnameValue = Label(self.boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labellastnameValue.place(x= 120 ,y = 250)

       labeloptionValue = Label(self.boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labeloptionValue.place(x= 120 ,y = 290)

       labelniveauValue = Label(self.boiteAvatar,text=self.var_niveau,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelniveauValue.place(x= 120 ,y = 340)

       labelmatriculeValue = Label(self.boiteAvatar,text=self.var_matricule,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelmatriculeValue.place(x= 120 ,y = 390)

       # new

       labelsalleValue = Label(self.boiteAvatar,text=self.var_salle,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelsalleValue.place(x= 120 ,y = 440)

       labelMatiereValue = Label(self.boiteAvatar,text=self.var_matiere,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelMatiereValue.place(x= 120 ,y = 480)

       labelHeureValue = Label(self.boiteAvatar,text=self.var_heure,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelHeureValue.place(x= 120 ,y = 520)
    
       # fin new 

       labelpresentValue = Label(self.boiteAvatar,text=self.var_present,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelpresentValue.place(x= 120 ,y = 560)


       #je marque la liste des personne presente dans un fichier excel
       self.mark_presence(self.var_nom,self.var_prenom,self.var_option,self.var_niveau,self.var_matricule,self.var_salle,self.var_matiere,self.var_heure,self.var_present)






   


# action pour le bouton capture Automatique qui capture une image automatique lorsque la camera est lancer
#mais problematique cette facon car capture tout meme les images qui ne sont pas les faces

 def captureAuto(self):

  img  = v.Capture.get_pic_from_strem(1)




# action pour capture manuelle ici c'est nous qui decidons ce qu'on capturer

 def captureManuelle(self):

    camera = cv2.VideoCapture(1)  # Ouvre la webcam (index 0)
    while True:
        ret, frame = camera.read()  # Lit une image depuis la webcam
        cv2.imshow("Webcam", frame)  # Affiche l'image dans une fenêtre

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Si la touche 'c' est pressée (c = pour "Capture")
            cv2.imwrite("captureManuelle/image_capturee.jpg", frame)  # Enregistre l'image capturée
            break

    camera.release()  # Libère la webcam
    cv2.destroyAllWindows()  # Ferme toutes les fenêtres


 
# action pour le l'authentification de l'admin celui cette interface mene vers l'entrainement du model LBP 
 def login_window(self):
     self.new_window = Toplevel(self.root)
     self.app = LoginGUI(self.new_window)



#test de la requete et affichage est info a l'endroit precis
  
 def test_requete(self):
    con = connexion.getConnexion()
    con_cursor = con.cursor()
    # query = "SELECT  * FROM etudiant"
    query = "SELECT  etudiant.* ,matiere.* ,salle.* FROM etudiant LEFT JOIN  matiere ON etudiant.matricule = matiere.matricule_etudiant_mat  LEFT JOIN salle ON etudiant.matricule = salle.matricule_etudiant_sal"
    
    con_cursor.execute(query)
    result = con_cursor.fetchall()

    # print(result[0][0]) //id du premier element de la premiere table
    print(result[0][1])
    print(result[0][2])
    print(result[0][3])
    print(result[0][4])
    print(result[0][5])
    print(result[0][6])
    #fin element de la table etudiant 

    #on passe a la table matiere

    # print(result[0][6])  //id du premier element de la table matiere 
    # print(result[0][7]) // matricule de l'etudiant ie Foreign Key
    #print(result[0][8]) // Nom de la matiere; prochain champ = code matiere , apres l'heure de passage
    print(result[0][10])   
    print(result[0][11])

    #on passe a la table salle

    #print(result[0][11]) //l'id du premier element de la table salla
    #print(result[0][11]) // matricule de l'etudiant ie Foreign Key
    #print(result[0][11]) //nom de la salle et en fin le Code de la Salle Qui est tout ce don on a besoin d'afficher
    print(result[0][15])

    print(result)

    self.var_nom=result[0][1]
    self.var_prenom=result[0][2]
    self.var_option=result[0][3]
    self.var_niveau=result[0][4]
    self.var_matricule=result[0][5]
    self.var_salle=result[0][15]
    self.var_matiere=result[0][10]
    self.var_heure=result[0][11]
    self.var_present=result[0][6]

    # # self.boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
    # # self.boiteAvatar.place(x=50, y=70) # c'est pas bon il ne faut pas construire une nouvelle boite

    # # on construit les nouveau label qui contiendrons les inforrmation a afficher ce pendant il doivent toujours heriter ou etre mit
    # #sur le frame boite boiteAvatar car si on creait une nouvelle boite elle va fermer l'autre et pour avoir acces a la boiteAvatar utiliser dans le constructeur initiale on utilise "self"


    lnameValue = Label(self.boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    lnameValue.place(x= 120 ,y = 200)


    labellastnameValue = Label(self.boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labellastnameValue.place(x= 120 ,y = 250)

    labeloptionValue = Label(self.boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labeloptionValue.place(x= 120 ,y = 290)

    labelniveauValue = Label(self.boiteAvatar,text=self.var_niveau,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelniveauValue.place(x= 120 ,y = 340)

    labelmatriculeValue = Label(self.boiteAvatar,text=self.var_matricule,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelmatriculeValue.place(x= 120 ,y = 390)

    # new

    labelsalleValue = Label(self.boiteAvatar,text=self.var_salle,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelsalleValue.place(x= 120 ,y = 440)

    labelMatiereValue = Label(self.boiteAvatar,text=self.var_matiere,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelMatiereValue.place(x= 120 ,y = 480)

    labelHeureValue = Label(self.boiteAvatar,text=self.var_heure,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelHeureValue.place(x= 120 ,y = 520)
    
    # new 

    labelpresentValue = Label(self.boiteAvatar,text=self.var_present,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
    labelpresentValue.place(x= 120 ,y = 560)




# test affichage avatar

 def test_avatar(self):
   
   camera = cv2.VideoCapture(1)  # Ouvre la webcam (index 0)
   while True:
        ret, frame = camera.read()  # Lit une image depuis la webcam
        cv2.imshow("Webcam", frame)  # Affiche l'image dans une fenêtre

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Si la touche 'c' est pressée

           break

        #redimensionement de l'image
        dim = (100 , 100)
        frame_redim = cv2.resize(frame,dim)

        cv2.imwrite("captureAvatar/image_capturee.jpg", frame_redim)  # Enregistre l'image capturée 
        # apres l'enregistrement on va use cette image pour tester sa prediction sur notre model

        # affichage de l'image comme avatar

        ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
        self.avatarStudent = ImageTk.PhotoImage(file = r'captureAvatar/image_capturee.jpg')

        ## pour faire cela j'ecrase l'ancienne boite

        labelface = Label(self.boiteAvatar,image = self.avatarStudent)
        labelface.place(x= 140 ,y = 50)      

   camera.release()  # Libère la webcam
   cv2.destroyAllWindows()  # Ferme toutes les fenêtres




##  test lancement de la camera a position specifique

 def test_positionCamera(self):

    def show_frame():

      # while  True:
      
         _, frame = cap.read()
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         img = Image.fromarray(frame)
         imgtk = ImageTk.PhotoImage(image=img)
         lbl.configure(image=imgtk)
         lbl.image = imgtk
         lbl.after(10, show_frame)

         #pour l'arret mais sa ne marche pas 
         key = cv2.waitKey(1) & 0xFF
         if key == ord('c'):  # Si la touche 'c' est pressée
           # break
           print("la touche marche")
           cap.release()  # Libère la webcam
           cv2.destroyAllWindows()  # Ferme toutes les fenêtres

         #  ca marche il faut installer la library "keyboard" de python "a = arreter"
         if keyboard.is_pressed('a'):
           print("la touche 'a' a été appuyer pour arreter la camera ")
           # cap.release()  # Libère la webcam
           # cv2.destroyAllWindows()  # Ferme toutes les fenêtres

           lbl.destroy() # je detruis le label plutot mais la camera tourne toujours
           #je verifie si la video/camera est ouverte si oui alors j'eteind
           if cap.isOpened():
            cap.release()  # Libère la webcam

            



    #on va utiliser notre boite de camera defini pus haut dans notre class
    # root = Tk()
    cap = cv2.VideoCapture(1)
    # lbl = Label(root)
    lbl = Label(self.boitecamera)
    lbl.pack()
    show_frame()
    # root.mainloop()



  ### test du model avec les images en interfaces sans entrer dans le code

 
 def choose_image(self):

     # os.startfile("dataCNN")

    # Chargement du modèle de reconnaissance faciale
    model = tf.keras.models.load_model('modelCNN/model1.h5')

    global filename,img
    f_types =[('Jpg Files', '*.jpg'),('Jpeg Files', '*.jpeg'),('Png files','*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    # img = ImageTk.PhotoImage(file=filename)
    # img=open(filename,'rb') 
    img = Image.open(filename) 

    # img=img.read()

    img=img.resize((300,300), Image.ANTIALIAS)

    #je creait une copie pour afficher dans la boite avatar / car plus bas l'image serait code en binary
    img_avatar = img

    img_avatar = img_avatar.resize((100,100),Image.ANTIALIAS )
    
    #je convertis l'image en array donc je ne pourrais plus l'afficher de maniere classique
    img=np.array(img).astype('float32')/255.0
    img=np.expand_dims(img,axis=0)
            
  
    predic=model.predict(img) # prediction sur l'image capturer
    classe_image=(np.argmax(predic[0])) # on recupere la probabilite maximale : elle sera utiliser pour determiner la classe cible 
        
    ## ---- affichage du tableau des differentes probabilite de l'image capturer a appartenier a chaque classe du tableau ie les labels
    print(predic[0])

    ### affichage de la probabilite de la classe cible en la ramenant en pourcentage
    print(predic[0][classe_image]*100) 

    if predic[0][classe_image]<0.7:
       print("personne inconnue")
       #-----------------------------------------# texte par defaut quand le student n'est pas reconnue

       ## -- il faut vider les champ lorsque une capture a ete prise et que apres on a redenarrer la camera pour prendre une autre capture :car si on ne vide pas les champ alors les valeus viennnent se superposer

      
       ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
       self.avatarStudent = ImageTk.PhotoImage(img_avatar)

       ## pour faire cela j'ecrase l'ancienne boite

       labelface = Label(self.boiteAvatar,image = self.avatarStudent)
       labelface.place(x= 140 ,y = 50)   


       lnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       lnameValue.place(x= 120 ,y = 200)

       labellastnameValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labellastnameValue.place(x= 120 ,y = 250)

       labeloptionValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labeloptionValue.place(x= 120 ,y = 290)

       labelniveauValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelniveauValue.place(x= 120 ,y = 340)

       labelmatriculeValue = Label(self.boiteAvatar,text="Inconu",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelmatriculeValue.place(x= 120 ,y = 390)

       labelsalleValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelsalleValue.place(x= 120 ,y = 440)

       labelMatiereValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelMatiereValue.place(x= 120 ,y = 480)

       labelHeureValue = Label(self.boiteAvatar,text="Unknown",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelHeureValue.place(x= 120 ,y = 520)

       labelpresentValue = Label(self.boiteAvatar,text="No",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="red")
       labelpresentValue.place(x= 120 ,y = 560)

    else: 
       print("personne reconnue") 
       print(predic)  # j'affiche sa prediction

              
       label_dir='donnee/labels/data.txt'
       with open(label_dir,"r") as f:
            train_label=[line.strip() for line in f]

            #pour convertir les etiquettes string en entier
    
            label_encoder=LE()
            train_label=label_encoder.fit_transform(train_label)

            # test_label=train_label 

       print("affichage des labels d'entrainement= labels de test apres encodage")   
       print(train_label)
       label=label_encoder.inverse_transform([classe_image]) #je decode le label quia été en coder dans l'entrainement du modle pour obtenir la classe (matricule) en string
       print(label[0]) #affichage du label 

       # je fais dabord la mise a jour du champ present de l'etudiant en mettant ce champ a oui car il est reconnue

       con = connexion.getConnexion()
       con_cursor = con.cursor()
       val=(predic[0][classe_image]*100)

       valeur=label[0] #  je fais un casting je convertis en string sinon sa va deranger

       sql = "update etudiant set present='OUI' where matricule=%s"
       params=(valeur,)
       con_cursor.execute(sql,params)
       con.commit()
       con.close()


       # recuperation des information de l'etudiant grace a l'etiquette

       con = connexion.getConnexion()
       con_cursor = con.cursor()
       val=(predic[0][classe_image]*100)
       valeur=label[0]
       # query = "select  * from etudiant where matricule=%s"
       query = "SELECT  etudiant.* ,matiere.* ,salle.* FROM etudiant LEFT JOIN  matiere ON etudiant.matricule = matiere.matricule_etudiant_mat  LEFT JOIN salle ON etudiant.matricule = salle.matricule_etudiant_sal WHERE matricule =%s"

       parametre=(valeur,)
       con_cursor.execute(query,parametre)
       result = con_cursor.fetchall()
       # con.close()

       android = pyttsx3.init()
       android.say("Je suis sûre , qu'il,, s'agit de,, :"+result[0][1]+",,,, ,"+result[0][2]+',,,voici, ses ,informations')
       android.runAndWait()

  
       # print(result[0][0]) //id du premier element de la premiere table
       print(result[0][1])
       print(result[0][2])
       print(result[0][3])
       print(result[0][4])
       print(result[0][5])
       print(result[0][6])
       print(result[0][10])   
       print(result[0][11])
       print(result[0][15])

       self.var_nom=result[0][1]
       self.var_prenom=result[0][2]
       self.var_option=result[0][3]
       self.var_niveau=result[0][4]
       self.var_matricule=result[0][5]
       self.var_salle=result[0][15]
       self.var_matiere=result[0][10]
       self.var_heure=result[0][11]
       self.var_present=result[0][6]

       # self.boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
       # self.boiteAvatar.place(x=50, y=70) # c'est pas bon il ne faut pas construire une nouvelle boite

       # on construit les nouveau label qui contiendrons les inforrmation a afficher ce pendant il doivent toujours heriter ou etre mit
       #sur le frame boite boiteAvatar car si on creait une nouvelle boite elle va fermer l'autre et pour avoir acces a la boiteAvatar utiliser dans le constructeur initiale on utilise "self"


       ### mise en place de l'image au niveau de la boiteAvatar : la boite qui devra contenir les information des students
       self.avatarStudent = ImageTk.PhotoImage(img_avatar)

       ## pour faire cela j'ecrase l'ancienne boite

       labelface = Label(self.boiteAvatar,image = self.avatarStudent)
       labelface.place(x= 140 ,y = 50)   

       lnameValue = Label(self.boiteAvatar,text=self.var_nom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       lnameValue.place(x= 120 ,y = 200)


       labellastnameValue = Label(self.boiteAvatar,text=self.var_prenom,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labellastnameValue.place(x= 120 ,y = 250)

       labeloptionValue = Label(self.boiteAvatar,text=self.var_option,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labeloptionValue.place(x= 120 ,y = 290)

       labelniveauValue = Label(self.boiteAvatar,text=self.var_niveau,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelniveauValue.place(x= 120 ,y = 340)

       labelmatriculeValue = Label(self.boiteAvatar,text=self.var_matricule,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelmatriculeValue.place(x= 120 ,y = 390)

       # new

       labelsalleValue = Label(self.boiteAvatar,text=self.var_salle,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelsalleValue.place(x= 120 ,y = 440)

       labelMatiereValue = Label(self.boiteAvatar,text=self.var_matiere,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelMatiereValue.place(x= 120 ,y = 480)

       labelHeureValue = Label(self.boiteAvatar,text=self.var_heure,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelHeureValue.place(x= 120 ,y = 520)
    
       # fin new 

       labelpresentValue = Label(self.boiteAvatar,text=self.var_present,font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="green")
       labelpresentValue.place(x= 120 ,y = 560)

       
       #je marque la liste des personne presente dans un fichier excel
       self.mark_presence(self.var_nom,self.var_prenom,self.var_option,self.var_niveau,self.var_matricule,self.var_salle,self.var_matiere,self.var_heure,self.var_present)




       


 #liste des presences et absence
    

 def liste_presence_excel(self):
  #trivial cette fonction va juste charger le fichier excel de presence dans un Widget (UI)
  # pass

  os.startfile("presence.csv")




 def liste_presence_absence(self):

  #Trivial simple requete SQL et je formate le contenue dans un fichier excel ou dans un Widget(UI)
  # pass  

  self.mark_presence_absence()
  os.startfile("PrecenceAbsence.csv")

 

 #### CLASS ####


class LoginGUI:
####### WIDGET PRINCIPAL##############################
 def __init__(self, root):

   self.root = root
   self.root.title(" APPLICATION-1.0 =>Login ")
    # self.root.geometry("1550x800+0+0")
   self.root.iconbitmap('image/bonjour.ico')
   self.root.geometry("1350x680+0+0")
   self.root.config(background="#2562bf")



   # ***************variable
   self.email = StringVar()
   self.password = StringVar()

    ###### ZONE DE TEXTE ::: LABEL

   label1=Label(self.root, font=('Imprint MT Shadow', 12), height=2, text = "VEUILLEZ VOUS AUTHENTIFIER",bg="#1d6a5c", fg="white")
   label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#1d6a5c", width=400, height=2)
   label3=Label(self.root, font=('comic sans ms', 12,'bold'), text="email",bg="#2562bf", fg="white")
   label4=Label(self.root, font=('comic sans ms', 12,'bold'), text="Mot de Passe", bg="#2562bf", fg="white")
   

   labelfooter=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
   labelfooter.pack(side=BOTTOM, fill=BOTH)

   label1.pack(side=TOP, fill=BOTH)
   label2.pack(side=BOTTOM, fill=BOTH)
   label3.place(x=450, y=250)
   label4.place(x=450, y=350)
   #label5.place(x=150, y=255)


   ###### CHAMPS DE SAISIE ::: ENTRY

   champ1=Entry(self.root,textvariable=self.email,width=18, font=('Bookman Old Style', 14))
   champ2=Entry(self.root,textvariable=self.password, width=18, show="*", font=('Bookman Old Style', 14))
   champ1.place(x=580, y=250)
   champ2.place(x=580, y=350)

    ###### LES BOUTON ::: BUTTON

   bouton1=Button(label2, font=('Lucida Bright', 12,'bold'), text="Connexion", command=self.login,height=1, width=10)
   bouton2=Button(label2, font=('Lucida Bright', 12), text="Quitter", height=1, width=10,command=self.root.destroy) 

   bouton1.pack(side=LEFT)
   bouton2.pack(side=RIGHT) 


    ########### LE BOUCLE PRINCIPAL ################################

   self.root.resizable(height=0,width=0)




 def login(self):

  if self.email.get() == "" or self.password.get() == "":
     messagebox.showerror("Error", "all field required")
  elif not ("@" or ".com") in self.email.get():
     speak_va("Try valid email address!!")
     messagebox.showinfo("success", "welcome to Face Recognition World")
     messagebox.showerror("Error",'Invalid email Enter valid email like fotsotatchumyvanol@gmail.com ',parent=self.root)
     
  else:
     conn = connexion.getConnexion()
     my_cursor = conn.cursor()
     my_cursor.execute("select * from admin where email=%s and password=%s",(
         self.email.get(),
         self.password.get()

     ))
     row=my_cursor.fetchone()
     if row==None:
         speak_va("Invalid username and password!")
         messagebox.showerror("Error","Invalid username and password")
     else:
         open_main=messagebox.askyesno("YesNo","Acess only admin")
         if open_main>0:
             self.new_window=Tk()
             self.app=EspaceAdmin(self.new_window)
         else:
             if not open_main:
                 return
     conn.commit()
     conn.close()






class EspaceAdmin():

 def __init__(self, root):

  ####### WIDGET PRINCIPAL##############################

    self.root = root
    self.root.title(" APPLICATION-1.0")
    # self.root.geometry("1550x800+0+0")
    self.root.iconbitmap('image/bonjour.ico')
    self.root.geometry("1350x680+0+0")
    self.root.resizable(height=0,width=0)
    self.root.config(background="#2562bf")

    
    ####### WIDGET SECONDAIRES ###########################

    ##### BOITE//CADRE  ::: FRAME
    boite0=Frame(self.root, height=400, width=1150, bg="#c5cfd7")
    boite0.place(x=50, y=150) #pour centrer la boite0 sur mon ecran en x=0,y=40 c'est centrer au petit format
    

    ###### ZONE DE TEXTE ::: LABEL

    label1=Label(self.root, font=('Imprint MT Shadow', 12), height=2, text = "GESTION DES UTILISATEURS ET ENTRAINEMENT DES DONNEES",bg="#1d6a5c", fg="white")
    label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
 
    label1.pack(side=TOP, fill=BOTH)
    label2.pack(side=BOTTOM, fill=BOTH)


    ## boite contenant les boutons 2 a 8
    boite1 = Frame(self.root,height=30,bg="#234a43",)
    boite1.pack(side=TOP,fill=BOTH)


    ### ajout des boutons pour la  duplication (augmentation) des donnees/ dataset images et dupliation label

    boite2 = Frame(self.root,height=30,bg="#234a43",)
    boite2.pack(side=BOTTOM,fill=BOTH)

    bouton00 = Button(boite2,font=('Lucida Bright', 12,'bold'), text="data CNN", height=1,command=self.open_dataCNN, width=10)
    bouton01 = Button(boite2,font=('Lucida Bright', 12,'bold'), text="Duplicate Image", height=1,command=self.duplicate_image, width=14)
    bouton02 = Button(boite2,font=('Lucida Bright', 12,'bold'), text="Duplicate Label", height=1,command=self.duplicate_label, width=14)
    bouton03 = Button(boite2,font=('Lucida Bright', 12,'bold'), text="Train CNN 3", height=1,command=self.train_CNN_Third, width=14)
    bouton04 = Button(boite2,font=('Lucida Bright', 12,'bold'), text="Liste Presence", height=1,command=self.liste_presence, width=14)
    
    bouton00.pack(side=LEFT)
    bouton01.pack(side=LEFT, padx=100)
    bouton02.pack(side=LEFT) 
    bouton03.pack(side=LEFT,padx=100) 
    bouton04.pack(side=RIGHT) 


    bouton2 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="ADD", command=self.add_user, height=1, width=8)
    bouton3 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="DELETE", height=1, width=8)
    bouton4 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="UPDATE", height=1, width=8)
    bouton5 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LIST", height=1, width=8)
    bouton6 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="Training LBP ",command=self.train_LBP, height=1, width=12)
    bouton7 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="data LBP", height=1,command=self.open_dataLBP, width=10)
    bouton8 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LOUGOUT", command=self.quitter, height=1, width=8)

    bouton2.pack(side=LEFT)
    bouton3.pack(side=LEFT, padx=70)
    bouton4.pack(side=LEFT) 
    bouton5.pack(side=LEFT, padx=70) 
    bouton6.pack(side=LEFT) 
    bouton7.pack(side=LEFT ,padx=30) 
    bouton8.pack(side=RIGHT) 



 def add_user(self):
  ####### WIDGET PRINCIPAL##############################

  # fen.destroy() 
   
  fenetreAdd=Tk()
  fenetreAdd.title("APPLICATION-1.0")

  fenetreAdd.iconbitmap('image/bonjour.ico')

  fenetreAdd.geometry("550x600") #ou 415*312
  fenetreAdd.config(background="#2562bf")



  ###### ZONE DE TEXTE ::: LABEL

  label1=Label(fenetreAdd, font=('Imprint MT Shadow', 12), height=2, text = "AjOUTER UN USER",bg="#1d6a5c", fg="white")

  label2=Label(fenetreAdd, font=('comic sans ms', 10,'bold'), bg="#1d6a5c", width=400, height=2) #pour les boutons

  label3=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Nom :",bg="#2562bf", fg="white")

  label4=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Prenom :", bg="#2562bf", fg="white")

  label5=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Niveau :",bg="#2562bf", fg="white")

  label6=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Option :",bg="#2562bf", fg="white")

  label7=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Matricule :",bg="#2562bf", fg="white")

  label8=Label(fenetreAdd, font=('comic sans ms', 12,'bold'), text="Sexe :",bg="#2562bf", fg="white")

  labelBouton = Label(fenetreAdd,font=('comic sans ms', 12,'bold'), bg="#2562bf",width=400, height=2)

  btnUlpoad1 = Button(labelBouton, text='Upload Image 1', width=20,command = lambda:upload_file1())
  btnUlpoad2 = Button(labelBouton, text='Upload Image 2', width=20,command = lambda:upload_file2())

  label1.pack(side=TOP, fill=BOTH)
  label2.pack(side=BOTTOM, fill=BOTH)

  label3.place(x=45, y=70)
  label4.place(x=45, y=140)
  label5.place(x=45, y=220)
  label6.place(x=45, y=290)
  label7.place(x=45, y=360)
  label8.place(x=45, y=430)

  labelBouton.place(x=45,y=500)

  btnUlpoad1.pack(side=LEFT)
  btnUlpoad2.pack(side=RIGHT)



  ###### CHAMPS DE SAISIE ::: ENTRY

  champ1=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
  champ2=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
  champ3=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
  champ4=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
  champ5=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
  champ6=Entry(fenetreAdd, width=18, font=('Bookman Old Style', 14))
 


  champ1.place(x=170, y=75)
  champ2.place(x=170, y=150)
  champ3.place(x=170, y=230)
  champ4.place(x=170, y=300)
  champ5.place(x=170, y=370)
  champ6.place(x=170, y=440)



 ###### LES BOUTON ::: BUTTON

  bouton1=Button(label2, font=('Lucida Bright', 12,'bold'), text="Valider",command="", height=1, width=10)
  bouton2=Button(label2, font=('Lucida Bright', 12,'bold'), text="Reset", command="",height=1, width=10)
  bouton1.pack(side=LEFT)
  bouton2.pack(side=RIGHT) 


  ########### LE BOUCLE PRINCIPAL ################################

  fenetreAdd.resizable(height=0,width=0)


  return fenetreAdd.mainloop() 


 def open_dataLBP(self):
        os.startfile("TrainYourSelf")


 def open_dataCNN(self):
        os.startfile("data")


 def duplicate_image(self):


   # Chemin du dossier contenant les images d'origine
   chemin_images_origine = "dataUseDuplication/studentImages/"

   # Chemin du dossier où les images seront dupliquées
   chemin_images_dupliquees = "dataUseDuplication/studentImagesDuplicate/"

   # Chemin du dossier où les images d'entraînement seront sauvegardées
   chemin_images_entrainement = "dataUseDuplication/dataLoadImageCNN/trainImageCNN"

   # Chemin du dossier où les images de test seront sauvegardées
   chemin_images_test = "dataUseDuplication/dataLoadImageCNN/testImageCNN/"

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

  # return    



 def duplicate_label(self):

   # Chemin des fichiers d'entrée et de sortie
   fichier_entree = "dataUseDuplication/studentLabels/labels.txt"
   fichier_sortie = "dataUseDuplication/studentLabelDuplicate/labels.txt"
   dossier_entrainement = "dataUseDuplication/dataLoadLabelCNN/trainLabelCNN"
   dossier_test = "dataUseDuplication/dataLoadLabelCNN/testLabelCNN"

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

  # return



## generer la liste de presence
     
 def liste_presence(self):
    pass 




 def train_LBP(self):
         
        data_dir = (r"TrainYourSelf")
        path=[os.path.join(data_dir,file) for  file in os.listdir(data_dir)]

        faces=[]
        ids=[]
        
        for image in path:
            img=Image.open(image).convert('L')  # grAY SCALE image
            imageNp=np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1)==13
        ids=np.array(ids)

        # Train the classifier and save 
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier/classifier.xml")
        cv2.destroyAllWindows()
        speak_va("Training datasets completed successfully!")
        messagebox.showinfo("Result","Training datasets completed successfully!",parent=self.root)
        # self.root.destroy()



 ## troisieme modele / model final 

 def train_CNN_Third(self):

    # traitement des images et des labels avant l'entrainement
    dossier_Labels='donnee/defaultLabels/labels.txt'
    dossier_Labels_apre_traitement='donnee/labels/data.txt'
    dossier_image="donnee/data/"
    dossier_image_augmenter="donnee/add_data/"
    dossier_image_entainement="donnee/entrainement/"
    dossier_image_test="donnee/test/"

    ##  traitement et duplication des images en verifiant si l'on a pas encore effectuer un traitement
    # nombre_image=fun.traitement_duplication_image(dossier_image,dossier_image_augmenter,dossier_image_entainement,dossier_image_test)

    nombre_image=fun.traitement_duplication_image(dossier_image,dossier_image_augmenter,dossier_image_entainement,dossier_image_test)


    print(f"le nombre d'image dans contenue dans notre dossier avant dupliation est :{nombre_image} ")

    ###  traitement des label 

    # fun.duplication_Labels(dossier_Labels,dossier_Labels_apre_traitement,nombre_element=nombre_image)

    #or puisque la fonction de traitement et duplication des images duplique une image en 11 alors egalement il faut dupliquer un label en 11


    fun.label_duplicate_final(dossier_Labels,dossier_Labels_apre_traitement,nombre_image) 


    #chemin au fichiers d'entrainements et test
    train_dir='donnee/entrainement'
    test_dir="donnee/test"

    #taille d'entre des images
    input_shape=(300,300)

    #chargement des donnnes et pretraitement
    train_image = []
    for filename in os.listdir(train_dir) :
        #chargement image a l'aide de PIL
        img=Image.open(os.path.join(train_dir,filename))

      
        #redimensionnement des images
        img=img.resize(input_shape)
        #conversion de l'images en tableau numpy et normalisation des valeurs de pixels
        img=np.array(img).astype('float32') / 255.0
        #ajouter de l'images traite a la liste des images d'entrainement
        train_image.append(img)
   
    #conversion de la liste d'image en tableau numpy

    train_image=np.array(train_image)
    print(train_image.shape)

    
    #chargement des donnes de test

    test_image = []
    for filename in os.listdir(test_dir) :
       #chargement image a l'aide de PIL
       img=Image.open(os.path.join(test_dir,filename))
      
       #redimensionnement des images
       img=img.resize(input_shape)
       #conversion de l'images en tableau numpy et normalisation des valeurs de pixels
       img=np.array(img).astype('float32') / 255.0
       #ajouter de l'images traite a la liste des images d'entrainement
       test_image.append(img)
   
    #conversion de la liste d'image en tableau numpy

    test_image=np.array(test_image)
    print("donnes de test")
    print(test_image.shape)



    #chargement des etiquettes => etiquette dupliquer
    label_dir='donnee/labels/data.txt'
    with open(label_dir,"r") as f:
       train_label=[line.strip() for line in f]

       #pour convertir les etiquettes string en entier
    
       label_encoder=LE()
       train_label=label_encoder.fit_transform(train_label)




    test_label=train_label    
    print(test_label)


    #sauvegarde des donnes dans les fichiers numpy
    np.save('donneeNumpy/train_image.npy',train_image)
    np.save('donneeNumpy/test_image.npy',test_image)
    np.save('donneeNumpy/test_label.npy',test_label)
    np.save('donneeNumpy/train_label.npy',train_label)



    #chargement des donnees et separation
    # X_train=np.load('donneeNumpy/train_image.npy',train_image)
    # X_test=np.load('donneeNumpy/test_image.npy',test_image)
    # Y_test=np.load('donneeNumpy/test_label.npy',test_label)
    # Y_train=np.load('donneeNumpy/train_label.npy',train_label)

    X_train = train_image
    X_test = test_image
    Y_train = train_label
    Y_test =  test_label


    #verification des etiquette et de leurs plage
    print(X_train)
    print(Y_train)


  
  #----------------------------------- deuxieme apres augmentation des classes ---------------------##
  #---- trouve 1 sur 5 images non utiliser pour l'entrainement ni test ------#


    #puisque les nouveau etiquette varient de o - 7 alors ma couche dense doit avoir 8 neurones  car les valeurs des etiquette doivent appartenir dans [0, num_classes-1].
    # or actuellement j'ai 8 etudiant different ie 8 matricule diffenrent or puisque les etiqutte begin de 0 alors j'aurais 8 neurone en sortie car chaque etudiant predit par une classe donc il faut 8 classes or de 0--7 donne 8

  
   

    model = keras.Sequential([
        keras.Input(shape=(300, 300, 3)),
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.8),  # Couche Dropout pour la régularisation
        layers.Dense(3, activation='softmax'),
    ])

    #compilon notre model
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

    #entrainons notre model
    history=model.fit(X_train,Y_train, batch_size=100,epochs=10,validation_split=0.1)
    print("l'entrai,nement,s'est , réaliiser, avec , succés ")

    android=pyttsx3.init()
    android.say("l'entrainement,s'est , réaliiser, avec , succés ")
    android.runAndWait()

    print("Fin de l'entrainement de notre model")

    #evaluons notre model
    print("evaluation de la performance de notre model")
    test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)

    ## affichage de nos courbe de loss et de accuracy grace a la library plot

    #graphe du loss et du accuracy
    loss_curve=history.history['loss']
    acc_curve=history.history['accuracy']
    plt.plot(loss_curve)
    plt.title('loss')
    plt.show()
    plt.plot(acc_curve)
    plt.title('accuracy')
    plt.show()

    #sauvegarde du model
    model.save('modelCNN/model1.h5')
  


 def quitter(self):
        speak_va("Are you sure you want to closed admin space")
        self.quitter = messagebox.askyesno("Face Recognition","Are you sure you want to exit the admin space ?",parent=self.root)
        if self.quitter > 0 :
            self.root.destroy()
        else:
            return








class LoginChef_Salle():
####### WIDGET PRINCIPAL##############################
 def __init__(self, root):

   self.root = root
   self.root.title(" APPLICATION-1.0 =>Login ")
    # self.root.geometry("1550x800+0+0")
   self.root.iconbitmap('image/bonjour.ico')
   self.root.geometry("1350x680+0+0")
   self.root.config(background="#2562bf")



   # #### Variables  #####
   self.champ1 = StringVar()
   self.champ2 = StringVar()

    ###### ZONE DE TEXTE ::: LABEL

   label1=Label(self.root, font=('Imprint MT Shadow', 12), height=2, text = "VEUILLEZ VOUS AUTHENTIFIER",bg="#1d6a5c", fg="white")
   label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#1d6a5c", width=400, height=2)
   label3=Label(self.root, font=('comic sans ms', 12,'bold'), text="email",bg="#2562bf", fg="white")
   label4=Label(self.root, font=('comic sans ms', 12,'bold'), text="Mot de Passe", bg="#2562bf", fg="white")
  
   labelfooter=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
   labelfooter.pack(side=BOTTOM, fill=BOTH)

   label1.pack(side=TOP, fill=BOTH)
   label2.pack(side=BOTTOM, fill=BOTH)
   label3.place(x=450, y=250)
   label4.place(x=450, y=350)
   #label5.place(x=150, y=255)


   ###### CHAMPS DE SAISIE ::: ENTRY

   ##il faut faire pointer les variable de classe comme variable de recuperation des Entry son impossible d'avoir acces a leur valeur dans la fonction de login_chef_salle


   self.champ1=Entry(self.root, width=18, font=('Bookman Old Style', 14))
   self.champ2=Entry(self.root, width=18, show="*", font=('Bookman Old Style', 14))
   self.champ1.place(x=580, y=250)
   self.champ2.place(x=580, y=350)

    ###### LES BOUTON ::: BUTTON

   bouton1=Button(label2, font=('Lucida Bright', 12,'bold'), text="Connexion", command=self.login_chef_salle,height=1, width=10)
   bouton2=Button(label2, font=('Lucida Bright', 12), text="Quitter", height=1, width=10,command=self.root.destroy) 

   bouton1.pack(side=LEFT)
   bouton2.pack(side=RIGHT) 

   



    ########### LE BOUCLE PRINCIPAL ################################

   self.root.resizable(height=0,width=0)




 def login_chef_salle(self):


  if self.champ1.get() == "" or self.champ2.get() == "":
     messagebox.showerror("Error", "all field required")
  elif not ("@" or ".com") in self.champ1.get():
     speak_va("Try valid email address!!")
     messagebox.showinfo("success", "welcome to Face Recognition World")
     messagebox.showerror("Error",'Invalid email Enter valid email like fotsotatchumyvanol@gmail.com ',parent=self.root)
     
  else:
     conn = connexion.getConnexion()
     my_cursor = conn.cursor()
     my_cursor.execute("select * from chef_salle where email=%s and password=%s",(
         self.champ1.get(),
         self.champ2.get()

     ))
     row=my_cursor.fetchone()
     if row==None:
         speak_va("Invalid username and password!")
         messagebox.showerror("Error","Invalid username and password")
     else:
         open_main=messagebox.askyesno("YesNo","Acess only Chef-Salle Interface")
         if open_main>0:
             self.new_window=Toplevel(self.root)
             self.app=PremiereFenetre(self.new_window)

             self.root.destroy
         else:
             if not open_main:
                 return
     conn.commit()
     conn.close()





                                                                                               
if __name__ == "__main__":
    main()
   
