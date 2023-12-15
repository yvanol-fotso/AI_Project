# import tkinter as tk

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


# import adminGestion as modul
import connexionBd as connexion


import cv2
import pyttsx3

import re
import numpy as np


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 is for female voice and 0 is for male voice



def main():
    win = Tk()
    app = PremiereFenetre(win)
    win.mainloop()


def speak_va(transcribed_query):
    engine.say(transcribed_query)
    engine.runAndWait()






# chargement par defaut du fichier haracast

# ..................load predefined data  face forntal from opencv.............
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


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
   self.root.iconbitmap('bonjour.ico')
   self.root.geometry("1250x650")
   self.root.resizable(height=0,width=0)
   self.root.config(background="#2562bf")



    # pour les images png
    # root.tk.call('wm','iconphoto',root._w,tk.PhotoImage(file='1.png')) 


   # pour tout les type d'images
   # root.iconphoto(False, tk.PhotoImage(file='3.jpg'))



   # creation de la boite gauche pour affichage des info du user

   bgframe = PhotoImage(file = '4.png')

   boiteAvatar=Frame(self.root, height=500, width=400, bg="#c5cfd7")
   boiteAvatar.place(x=50, y=70) 


   labelface = Label(boiteAvatar,image = bgframe)
   labelface.place(x= 140 ,y = 50)

   labelname = Label(boiteAvatar,text="Nom:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelname.place(x= 50 ,y = 230)

   labellastname = Label(boiteAvatar,text="Prenom:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labellastname.place(x= 50 ,y = 280)

   labeloption = Label(boiteAvatar,text="Option:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labeloption.place(x= 50 ,y = 320)

   labelniveau = Label(boiteAvatar,text="Niveau:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelniveau.place(x= 50 ,y = 370)

   labelmatricule = Label(boiteAvatar,text="Matricule:",font=('comic sans ms', 10,'bold'),bg="#c5cfd7",fg="white")
   labelmatricule.place(x= 50 ,y = 420)






   # boite droite pour le demarrage de la camera
   boitecamera = Frame(self.root, height=500, width=600, bg="#2562bf",bd=1)
   boitecamera.place(x=600, y=70) 

   # bgcamera = PhotoImage(file = 'camera.png')
   bgcamera = PhotoImage(file = 'camera1.png')


   labelcamera = Label(boitecamera,image= bgcamera)
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



   # ajout des options a la menuBar

   menuTraining = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="No Student", menu=menuTraining)
   menuTraining.add_command(label="Training Y're Self", command=self.trainYouSelf)
   menuTraining.add_separator()
   menuTraining.add_command(label="Reconize",command=self.face_recog)

   # ajout des options a la menuBar

   menuAdmin = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Admin", menu=menuAdmin)
   menuAdmin.add_command(label="Login",command=self.login_window)

   # ajout des options a la menuBar

   menuMeethod = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Method", menu=menuMeethod)

   var_case1 = IntVar()
   var_case2 = IntVar()

   menuMeethod.add_checkbutton(label="Haar",variable = var_case1,command="")
   menuMeethod.add_separator()
   menuMeethod.add_checkbutton(label="LBP",variable = var_case2,command="")

   #on peut controler l etat de la case a cocher en interogeant la variable qui retourne 1 si oui 0 sinon on peut aussi la lie avec une command
   print(var_case1.get())
   print(var_case2.get())


   menuquit = Menu(menubar,tearoff=0) 
   menubar.add_cascade(label="Leave", menu=menuquit)
   menuquit.add_command(label="leave",command=self.root.destroy)


   label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
   label2.pack(side=BOTTOM, fill=BOTH)







  ##### FONCTION ######
 def login_window(self):
     self.new_window = Toplevel(self.root)
     self.app = LoginGUI(self.new_window)


  ## lancement de la camera ##



 def face_cropped(img):

   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces=face_classifier.detectMultiScale(gray,1.3,5)

   for (x,y,w,h) in faces:
       face_cropped=img[y:y+h,x:x+w]
       return face_cropped


 # lancement de la camera 

 def lancerCamera(self):

 
  cam = cv2.VideoCapture(0)

  while (True):
      ret,frame = cam.read()

      # display the frame
      cv2.imshow('Camera ', frame)
      # wait for 100 miliseconds
      if cv2.waitKey(100) & 0xFF == ord('q'):
          break

  cam.release()
  cv2.destroyAllWindows()

  	 

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



   ### deuxieme methode pour l'allumage de la camera


    
   # Arreter la camera

 def  offCamera(self):
	
   if cv2.VideoCapture(0):
   	cv2.destroyAllWindows() 
   speak_va("Camera is closed succefull.")
   messagebox.showinfo("Result","Camera is closed succefully!!!",parent=self.root)
   return 1
	



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



   # code pour la prise des images des personne non etudiant desirant essayer

 def trainYouSelf(self):

  assure_path_exists("data/")
  name = "user"

  cam = cv2.VideoCapture(0)
  harcascadePath = "haarcascade_frontalface_default.xml"
  detector = cv2.CascadeClassifier(harcascadePath)
  sampleNum = 0
  while (True):
      ret, img = cam.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.05, 5)
      for (x, y, w, h) in faces:
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          # incrementing sample number
          sampleNum = sampleNum + 1
          # saving the captured face in the dataset folder TrainingImage
          cv2.imwrite("data/ " + name + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])

          cv2.putText(img,str(sampleNum),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

          # display the frame
          cv2.imshow('Taking Images', img)
      # wait for 100 miliseconds
      if cv2.waitKey(100) & 0xFF == ord('q'):
          break
      # break if the sample number is morethan 100
      elif sampleNum > 100:
          break
  cam.release()
  cv2.destroyAllWindows() 
  speak_va("Generation of Data Set completed.")
  messagebox.showinfo("Result","Generation of data set completed!!!",parent=self.root)

  

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
                    cv2.putText(img,f"id: {i}",(x,y-75),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"Roll:{r}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"Name:{n}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"Department:{d}",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
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
        
        faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

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
    

 


class LoginGUI:
####### WIDGET PRINCIPAL##############################
 def __init__(self, root):

   self.root = root
   self.root.title(" APPLICATION-1.0 =>Login ")
    # self.root.geometry("1550x800+0+0")
   self.root.iconbitmap('bonjour.ico')
   self.root.geometry("550x300")
   self.root.config(background="#2562bf")



   # ***************variable
   self.email = StringVar()
   self.password = StringVar()

    ###### ZONE DE TEXTE ::: LABEL

   label1=Label(self.root, font=('Imprint MT Shadow', 12), height=2, text = "VEUILLEZ VOUS AUTHENTIFIER",bg="#1d6a5c", fg="white")
   label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#1d6a5c", width=400, height=2)
   label3=Label(self.root, font=('comic sans ms', 12,'bold'), text="email",bg="#2562bf", fg="white")
   label4=Label(self.root, font=('comic sans ms', 12,'bold'), text="Mot de Passe", bg="#2562bf", fg="white")
   #label5=Label(self.root, font=('Jokerman', 12), text='Bonne\nrentree\ndu\nsemestre2\n2021', bg="green", fg="yellow",height=5, width=20)
   label1.pack(side=TOP, fill=BOTH)
   label2.pack(side=BOTTOM, fill=BOTH)
   label3.place(x=45, y=80)
   label4.place(x=45, y=140)
   #label5.place(x=150, y=255)


   ###### CHAMPS DE SAISIE ::: ENTRY

   champ1=Entry(self.root,textvariable=self.email,width=18, font=('Bookman Old Style', 14))
   champ2=Entry(self.root,textvariable=self.password, width=18, show="*", font=('Bookman Old Style', 14))
   champ1.place(x=170, y=85)
   champ2.place(x=170, y=145)

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
    self.root.iconbitmap('bonjour.ico')
    self.root.geometry("1250x650")
    self.root.resizable(height=0,width=0)
    self.root.config(background="#2562bf")

    
    ####### WIDGET SECONDAIRES ###########################

    ##### BOITE//CADRE  ::: FRAME
    boite0=Frame(self.root, height=300, width=1150, bg="#c5cfd7")
    boite0.place(x=50, y=150) #pour centrer la boite0 sur mon ecran en x=0,y=40 c'est centrer au petit format




    ###### ZONE DE TEXTE ::: LABEL

    label1=Label(self.root, font=('Imprint MT Shadow', 12), height=2, text = "GESTION DES UTILISATEURS",bg="#1d6a5c", fg="white")
    label2=Label(self.root, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
 
    label1.pack(side=TOP, fill=BOTH)
    label2.pack(side=BOTTOM, fill=BOTH)


    boite1 = Frame(self.root,height=30,bg="#234a43",)
    boite1.pack(side=TOP,fill=BOTH)


    bouton2 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="ADD", command=self.add_user, height=1, width=10)
    bouton3 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="DELETE", height=1, width=10)
    bouton4 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="UPDATE", height=1, width=10)
    bouton5 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LIST", height=1, width=10)
    bouton6 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="Training data",command=self.train_data, height=1, width=14)
    bouton7 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="data", height=1,command=self.open_data, width=14)
    bouton8 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LOUGOUT", command=self.quitter, height=1, width=10)

    bouton2.pack(side=LEFT)
    bouton3.pack(side=LEFT, padx=90)
    bouton4.pack(side=LEFT) 
    bouton5.pack(side=LEFT, padx=90) 
    bouton6.pack(side=LEFT) 
    bouton7.pack(side=LEFT) 
    bouton8.pack(side=RIGHT) 



 def add_user(self):
  ####### WIDGET PRINCIPAL##############################

  # fen.destroy() 
   
  fenetreAdd=Tk()
  fenetreAdd.title("APPLICATION-1.0")

  fenetreAdd.iconbitmap('bonjour.ico')

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


 def open_data(self):
        os.startfile("data")



 def train_data(self):
         
        data_dir = (r"data")
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
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        speak_va("Training datasets completed successfully!")
        messagebox.showinfo("Result","Training datasets completed successfully!",parent=self.root)
        # self.root.destroy()

  
 def quitter(self):
        speak_va("Are you sure you want to closed admin space")
        self.quitter = messagebox.askyesno("Face Recognition","Are you sure you want to exit the admin space ?",parent=self.root)
        if self.quitter > 0 :
            self.root.destroy()
        else:
            return

                                                                                               
if __name__ == "__main__":
    main()
   
