
import connexionBd as connexion

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
import login 

import cv2
import pyttsx3


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 is for female voice and 0 is for male voice


def main():
    win = Tk()
    app = loginGUI(win)
    win.mainloop()


def speak_va(transcribed_query):
    engine.say(transcribed_query)
    engine.runAndWait()



class LoginGUI():
####### WIDGET PRINCIPAL##############################
 def __init__(self, root):

   self.root = root
   self.root.title(" APPLICATION-1.0 =>Login ")
    # self.root.geometry("1550x800+0+0")
   self.root.iconbitmap('bonjour.ico')
   self.root.geometry("550x300")
   self.root.config(background="#2562bf")

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

   champ1=Entry(self.root, width=18, font=('Bookman Old Style', 14))
   champ2=Entry(self.root, width=18, show="*", font=('Bookman Old Style', 14))
   champ1.place(x=170, y=85)
   champ2.place(x=170, y=145)

    ###### LES BOUTON ::: BUTTON

   bouton1=Button(label2, font=('Lucida Bright', 12,'bold'), text="Connexion", command=login,height=1, width=10)
   bouton2=Button(label2, font=('Lucida Bright', 12), text="Quitter", height=1, width=10,command=self.root.destroy) 

   bouton1.pack(side=LEFT)
   bouton2.pack(side=RIGHT) 


    ########### LE BOUCLE PRINCIPAL ################################

   self.root.resizable(height=0,width=0)




 def login():

  if champ1.get() == "" or champ2.get() == "":
     messagebox.showerror("Error", "all field required")
  elif not ("@" or ".com") in champ1.get():
     speak_va("Try valid email address!!")
     messagebox.showinfo("success", "welcome to Face Recognition World")
     messagebox.showerror("Error",'Invalid email Enter valid email like fotsotatchumyvanol@gmail.com ',parent=self.root)
     
  else:
     conn = connexion.getConnexion()
     my_cursor = conn.cursor()
     my_cursor.execute("select * from admin where email=%s and pass=%s",(
         champ1.get(),
         champ2.get()

     ))
     row=my_cursor.fetchone()
     if row==None:
         speak_va("Invalid username and password!")
         messagebox.showerror("Error","Invalid username and password")
     else:
         open_main=messagebox.askyesno("YesNo","Acess only admin")
         if open_main>0:
             self.new_window=Toplevel(self.root)
             self.app=EspaceAdmin(self.new_window)
         else:
             if not open_main:
                 return
     conn.commit()
     conn.close()





                                                                                                 
if __name__ == "__main__":
    main()
    