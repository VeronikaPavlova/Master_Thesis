from tkinter import *
from tkinter import filedialog

root = Tk()
root.title("AAS and SS loading and plotting! :)")


def load_file():
    root.filename = filedialog.askopenfilenames(parent=root, title="Choose a file")
    my_label = Label(frame_load, text=root.filename).grid(row=1)


frame_load = LabelFrame(root, text="Load SS rosbag file")
frame_load.grid()

load_file_btn = Button(frame_load, text="Open file", command=load_file)
load_file_btn.grid(row=0)

root.mainloop()
