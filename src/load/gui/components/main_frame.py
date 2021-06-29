from tkinter import ttk

from gui.components.ss_frame import SSFrame
from gui.components.ass_frame import ASSFrame

from gui.components.menu_bar import MenuBar


class MainFrame(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.build()

    def build(self) -> None:
        """Build the UI"""
        # MenuBar(self)
        self.tab_control = ttk.Notebook(self)
        self.tab_control.grid(row=0, column=0)

        ss_frame = SSFrame(self)
        ass_frame = ASSFrame(self)

        self.tab_control.add(ss_frame, text= "   SS   ")
        self.tab_control.add(ass_frame, text= "  ASS   ")
