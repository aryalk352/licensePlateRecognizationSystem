import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from keras.preprocessing import image
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from keras.layers import Embedding, Dense, Input, Dropout, LSTM, Activation, Conv2D, Reshape, Average, Bidirectional, Flatten


# In[2]:


def uploadPhoto():
    #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    global filename
    filename = askopenfilename(title = "Select file",filetypes = (("png files","*.jpg"),("all files","*.*"))) # show an "Open" dialog box and return the path to the selected file
    
    im = Image.open(filename)
    im = im.resize((250, 250), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(im)
    
    labelIM = tk.Label(root, image = photo)
    labelIM.image = photo
    labelIM.pack()


# In[3]:


def create_dir (dirName): 
  try:
      # Create target Directory
      os.mkdir(dirName)
      print("Directory " , dirName ,  " Created ") 
  except FileExistsError:
      print("Directory " , dirName ,  " already exists")


# In[4]:


def remove_dir (dirName): 
  try:
      # Create target Directory
      shutil.rmtree(dirName)
      print("Directory " , dirName ,  " Removed ") 
  except FileExistsError:
      print("Directory " , dirName ,  " Not Found")


# In[5]:



src = 'C:/Users/Zhipeng/Pictures/Image/folder 1/'
demoP = src+'demo'
create_dir(demoP)


# In[6]:


RESNET=models.load_model('C:/Users/Zhipeng/Downloads/50-0.02.hdf5')
RESNET.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    


# In[7]:


def openFile():
    src = 'C:/Users/Zhipeng/Pictures/Image/folder 1/'
    demoP = src+'demo'
    remove_dir(demoP)
    create_dir(demoP)
    dst = demoP + '/1.jpg'
    shutil.copy(filename, dst)
    
    print("Submitted " + filename)
    
    Labels = ['9326871', '9332898', '9338446', '9338454', '9338462', '9338489', '9338497', '9338519', '9338527', '9338543', '9414649', '9416994', 'Ash', 'Jason', 'Leo', 'admars', 'ahodki', 'ajflem', 'ajones', 'ajsega', 'akatsi', 'ambarw', 'asheal', 'bplyce', 'cchris', 'ccjame', 'cferdo', 'cgboyc', 'cjcarr', 'cjdenn', 'cjsake', 'cmkirk', 'csanch', 'cshubb', 'cwchoi', 'dagran', 'dakram', 'dcbowe', 'dioann', 'djbirc', 'djhugh', 'djmart', 'dmwest', 'gdhatc', 'ggeorg', 'ggrego', 'gjhero', 'gjnorm', 'gmwate', 'gpapaz', 'gpsmit', 'gsreas', 'irdrew', 'jabins', 'jagrif', 'jcarte', 'jdbenm', 'jgloma', 'jlemon', 'jmedin', 'jrtobi', 'kaatki', 'kdjone', 'khchan', 'khughe', 'kjwith', 'lejnno', 'maasht', 'mberdo', 'mdpove', 'mefait', 'mhwill', 'miaduc', 'mjhans', 'mpetti', 'muthay', 'nahaig', 'namull', 'ndbank', 'ndhagu', 'nhrams', 'njmoor', 'npbour', 'npmitc', 'nrclar', 'nrrbar', 'nwilli', 'ohpark', 'pacole', 'pmives', 'pshurr', 'pspliu', 'ptnich', 'rarobi', 'rgharr', 'rgspru', 'rjlabr', 'rlocke', 'rmcoll', 'rmpugh', 'rnpwil', 'rrowle', 'rsanti', 'saduah', 'saedwa', 'sidick', 'sjbeck', 'skumar', 'smrobb', 'spletc', 'svkriz', 'swewin', 'swsmit', 'vpsavo', 'whussa', 'wjalbe']
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    demo_generator = test_datagen.flow_from_directory(
        src,  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 64X64
        batch_size=1,
        #seed = 42
        shuffle = False)
    
    
    preds = RESNET.predict_generator(demo_generator, steps = 1, workers = 1, verbose  = 2)
    predicted_class_indices=np.argmax(preds,axis=1)
    pred1 = predicted_class_indices[0]
    
    PREDICT = "Welcome {}".format(Labels[pred1])
    
    
    labelF = tk.Label(root, text = PREDICT)
    labelF.pack()
    

root = tk.Tk()

filename  = ""

topFrame = tk.Frame(root)
topFrame.pack(side = tk.TOP) # intergrate what has been initialised


button1 = tk.Button(topFrame, text = 'Upload Photo', command = uploadPhoto)
button1.pack(side = tk.LEFT)


button2 = tk.Button(topFrame, text='Run Model', command=openFile)
button2.pack(side = tk.RIGHT)

root.mainloop() # keep window open till closed