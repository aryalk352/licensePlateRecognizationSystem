from tkinter import *
from tkinter import filedialog
import tkinter.messagebox
import importlib, importlib.util
from PIL import ImageTk, Image
from keras.preprocessing import image
import MainProject as mp

mainProject = mp.MainProject();
root = Tk()
root.geometry("900x800+0+0")
root.title('License Plate Detection')

top = Frame(root,width=900,height = 40,bg = "SteelBlue1", relief=SUNKEN)
top.pack(side=TOP)

f1 = Frame(root,width=100,height = 760,bg = "powder blue", relief=SUNKEN)
f1.pack(side=LEFT)

f2 = Frame(root,width=800,height = 760,bg = "SteelBlue1", relief=SUNKEN)
f2.pack(side=RIGHT)


#def uploadfile():

def intro():
	lbl = Label(top, font=(None,40,'bold'),text = "Welcome to the License Plate Detector",fg="gray0",bd=40,anchor='w')
	lbl.pack()

         
def menuBar():
	menubar = Menu(root)
	root.config(menu = menubar)
	subMenu = Menu(menubar, tearoff =0)
	#menubar.add_cascade(label = 'File', menu = subMenu)
	#subMenu.add_command(label = 'Upload Image', command = uploadfile)
	#subMenu.add_command(label = 'Exit', command = root.destroy
	subMenu = Menu(menubar, tearoff = 0)
	menubar.add_cascade(label = 'Help', menu = subMenu)
	subMenu.add_command(label = 'About Us', command = aboutus)

def aboutus():
    tkinter.messagebox.showinfo('About License Plate detector', 'This the copy right with The Three Muskeeters group of Deep Learning')
	
def displayMsg(message):
    tkinter.messagebox.showinfo(message)
	
def addButtons():
	uploadButton = Button(f1, text = "upload image", command = showUpload)
	uploadButton.grid(row=0,column=0)
	startdetection = Button(f1, text = "Start Detection", command = buttonDetect)
	startdetection.grid(row=1,column=0)
        
def showUpload():
	global filename
	filename = filedialog.askopenfilename(title = "Select file",filetypes = (("png files","*.jpg"),("all files","*.*")))
	img = Image.open(filename)
	photo = ImageTk.PhotoImage(img)
	labelIM = Label(f2, image = photo)
	labelIM.image = photo
	labelIM.grid(row=0,column=0)

def buttonDetect():
	cropped_path = mainProject.get_license_plate(filename)
	img = Image.open(cropped_path)
	photo = ImageTk.PhotoImage(img)
	labelIM = Label(f2, image = photo)
	labelIM.image = photo
	labelIM.grid(row=2,column=0)
	license_plate_characters = mainProject.get_number_value(cropped_path)
	displayname = Label(f2, font=(None,40,'bold'),text = license_plate_characters,fg="gray0",bd=40,anchor='w')
	displayname.grid(row=3,column=0)

intro();
menuBar();
addButtons();
root.mainloop()




