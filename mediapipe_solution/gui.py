from tkinter import *
import cv2 
from PIL import ImageTk, Image

class GUI: 
    
    def __init__(self, title, size):
        #initialized the tkinter class
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(size)
        
    def create_frame(self, width, height, anchor, relx, rely, background="Black"):
        frame = Frame(self.root, bg=background, width=width, height=height)
        # frame.place(anchor=anchor, relx=relx, rely=rely)
        frame.pack(side= RIGHT)
        print("create_frame")
        return frame
    
    def create_labels(self, label_num, labels, anchor, relx, rely, x_spacing=0, y_spacing=0, create_entrybox_per_label=False):
        entry_labels = {}
        entry_boxes = {} 
        relx = relx
        rely = rely
        
        max_lable_spacing = len(max(labels, key=len))/100.0
        
        #go through the list of labels
        # intialize all of them
        for i in range(label_num):
            label = Label(self.root, text = labels[i] + ": ",
                    font = ("Montserrat", 15))
            
            # place the labels in anchor position like northwest or northeast
            # relx and rely is how much from the top left hand corner 
            # should the x and y position move by
            # default is zero 
            label.place(anchor=anchor, relx=relx, rely=rely)
            
            entry_labels[labels[i]] = label
            #create text part of gui where users can type in values
            if create_entrybox_per_label:
                entry_box = Text(self.root, font=("Montserrat", 20), height=1, width=10)
                entry_box.place(anchor=anchor, relx=relx+max_lable_spacing+0.02, rely=rely)
                
                entry_boxes[labels[i] + '_entrybox'] = entry_box
            
            relx += x_spacing
            rely += y_spacing
        #return dictionary  of labels and entry boxes
        # entry_labels -> threshold : 'threshold'
        # entry_boxes -> threshold_entrybox : Text<object>
        return entry_labels, entry_boxes
    
    
    def create_buttons(self, bottom_num, text, anchor, relx, rely, command=None, x_spacing=0, y_spacing=0):
        buttons = {}
        relx = relx 
        rely = rely
        
        #same initialization as labels
        for i in range(bottom_num):
            btn = Button(self.root, command=command, text=text[i])
            btn.place(anchor=anchor, relx=relx, rely=rely)
            
            buttons[text[i]+ ' button'] = btn
            
            rely += y_spacing
            relx += x_spacing
            
        return buttons

    