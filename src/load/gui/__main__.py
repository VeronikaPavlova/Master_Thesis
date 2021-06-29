"""Start GUI here"""

import tkinter as tk

from gui.components.plot_gui import PlotGui


def main():
    root = tk.Tk()
    PlotGui(root).grid()
    root.mainloop()


if __name__ == "__main__":
    main()
