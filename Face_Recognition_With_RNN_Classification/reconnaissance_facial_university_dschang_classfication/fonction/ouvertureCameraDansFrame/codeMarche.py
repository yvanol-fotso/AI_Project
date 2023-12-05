import cv2
from tkinter import *
from PIL import Image, ImageTk

def show_frame():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.configure(image=imgtk)
    lbl.image = imgtk
    lbl.after(10, show_frame)

root = Tk()
cap = cv2.VideoCapture(1)
lbl = Label(root)
lbl.pack()
show_frame()
root.mainloop()
