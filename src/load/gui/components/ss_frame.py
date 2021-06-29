import os
from tkinter import ttk, messagebox
from tkinter import *

import matplotlib.pyplot as plt

import rosbag
from load import read_rosbag

from gui.components.load_frame import LoadFrame

class SSFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.fig, self.axis, self.canvas = None, None, None
        self.plot_frame = Frame(self)
        self.build()

    def build(self):
        self.load_frame = LoadFrame(self)
        self.load_frame.grid(row=0, column=0, padx=100, pady=100)

        plot_btn = Button(self, text="Plot")
        plot_btn.grid(row=1, column=0)
        plot_btn.bind("<Button-1>", self.plot)

    def plot(self, event) -> None:
        # close all pyplots to prevent memory leak
        plt.close("all")

        self.plot_frame.destroy()
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1)

        # self.filename = self.load_frame.list_files.get(ACTIVE)
        self.filenames = [self.load_frame.path[idx] for idx in self.load_frame.list_files.curselection()]

        for full_path in self.filenames:
            self.read_rosbag(full_path)

    def read_rosbag(self, full_path):

        try:
            bag = rosbag.Bag(full_path)
        except:
            messagebox.showerror("RosBag Error", "Please only select rosbag files")

        path, file = os.path.split(full_path)
        n, label = read_rosbag.get_num_and_label(file)