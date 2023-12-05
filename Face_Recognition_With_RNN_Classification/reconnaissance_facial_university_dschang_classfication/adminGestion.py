####################################

from tkinter import*

from tkinter import filedialog
from PIL import Image, ImageTk



class EspaceAdmin():

 def __init__(self, root):

  ####### WIDGET PRINCIPAL##############################

    fenetre=Tk()
    fenetre.title("APPLICATION-1.0")

    fenetre.iconbitmap('bonjour.ico')

    fenetre.geometry("1250x650") 

    fenetre.resizable(height=0,width=0)

    fenetre.config(background="#2562bf")

    ####### WIDGET SECONDAIRES ###########################

    ##### BOITE//CADRE  ::: FRAME
    boite0=Frame(fenetre, height=300, width=1150, bg="#c5cfd7")
    boite0.place(x=50, y=150) #pour centrer la boite0 sur mon ecran en x=0,y=40 c'est centrer au petit format




    ###### ZONE DE TEXTE ::: LABEL

    label1=Label(fenetre, font=('Imprint MT Shadow', 12), height=2, text = "GESTION DES UTILISATEURS",bg="#1d6a5c", fg="white")
    label2=Label(fenetre, font=('comic sans ms', 10,'bold'), bg="#084172", text="© coprigth 2023 by yvanol fotso", width=400, height=2)
 
    label1.pack(side=TOP, fill=BOTH)
    label2.pack(side=BOTTOM, fill=BOTH)


    boite1 = Frame(fenetre,height=30,bg="#234a43",)
    boite1.pack(side=TOP,fill=BOTH)


    bouton2 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="ADD", command=self.add_user, height=1, width=10)
    bouton3 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="DELETE", height=1, width=10)
    bouton4 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="UPDATE", height=1, width=10)
    bouton5 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LIST", height=1, width=10)
    bouton6 = Button(boite1,font=('Lucida Bright', 12,'bold'), text="Training data", height=1, width=14)
    bouton = Button(boite1,font=('Lucida Bright', 12,'bold'), text="LOUGOUT", command=fenetre.destroy, height=1, width=10)

    bouton2.pack(side=LEFT)
    bouton3.pack(side=LEFT, padx=90)
    bouton4.pack(side=LEFT) 
    bouton5.pack(side=LEFT, padx=90) 
    bouton6.pack(side=LEFT) 
    bouton7.pack(side=RIGHT) 



 ########### LE BOUCLE PRINCIPAL ################################

 # return fenetre.mainloop() 

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




def upload_file1():
 global img
 f_types = [('Jpg Files', '*.jpg'),('PNG files','*.png')]
 filename = filedialog.askopenfilename(filetypes=f_types)
 #img = ImageTk.PhotoImage(file=filename)
 img=Image.open(filename)



def upload_file2():
 global img
 f_types = [('Jpg Files', '*.jpg'),('PNG files','*.png')]
 filename = filedialog.askopenfilename(filetypes=f_types)
 #img = ImageTk.PhotoImage(file=filename)
 img=Image.open(filename)