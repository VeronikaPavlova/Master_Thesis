import os
from tkinter import ttk, messagebox
from tkinter import *

from tkinter import filedialog


class LoadFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.loaded_files = {}
        self.path = []
        self.file= []
        self.build()

    def open_files(self):
        self.files = filedialog.askopenfilenames(parent=self, initialdir="/home/nika/git/Master_Thesis/src/data/experiment_data/",title="Select One or More Files")
        for f in self.files:
            path, file = os.path.split(f)
            end = os.path.splitext(file)[1]
            if end != ".bag":
                messagebox.showerror("Not a RosBag Error", "File " + str(file) + " is not a rosbag")
            elif f not in self.loaded_files:
                self.path.append(f)
                self.list_files.insert(END, file)

    def detete_entry(self):
        # get selected elements from list to delete
        selected_files = [self.list_files.get(idx) for idx in self.list_files.curselection()]
        for idx in self.list_files.curselection():
            self.list_files.delete(idx)
            del self.path[idx]
        print("delete")

    def build(self):

        self.list_files = Listbox(self, selectmode=MULTIPLE, width=50)
        self.list_files.grid(row=1)
        # values = self.list_files.get(ACTIVE)

        self.button_frame = ttk.Frame(self)

        ttk.Button(self.button_frame, text="Open File", command=self.open_files, width=10).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Delete", command=self.detete_entry, width=10).grid(row=0, column=1, padx=5)

        self.button_frame.grid(row=0)
