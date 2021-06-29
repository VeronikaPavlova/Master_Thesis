"""Core object for th application"""
from tkinter import font, ttk

from gui.components.main_frame import MainFrame


class PlotGui(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        # ========== Set UI ==========
        # title
        parent.title("AAS and SS Visualisation Tool")

        # font
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Arial")
        parent.option_add("*Font", "TkDefaultFont")

        # widget backgrounds / themes
        # ttk.Style().configure("TLabel", background="#FAFAFA")

        # TODO Add LogWindow?
        MainFrame(self).grid()