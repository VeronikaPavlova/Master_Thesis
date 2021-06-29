from tkinter import ttk
import tkinter as tk


class ASSFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.fig, self.axis, self.canvas = None, None, None
        self.plot_frame = tk.Frame(self)

