import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import pathlib
import matplotlib
from matplotlib import pyplot as plt

root = tk.Tk()
CURRENT = pathlib.Path(__file__).parent


class demo_gui():
    def __init__(self, root, models):
        # models should be a dictionary
        # name of the model: loaded model object
        self.models = models

        root.title("Image Colorization Demo")

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky="eswn")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.filename = tk.StringVar()
        self.pick_button = ttk.Button(
            root, text="Choose an Image File", command=self.open_file).grid(row=1, column=0, sticky="ew")
        ttk.Label(mainframe, textvariable=self.filename).grid(
            row=1, column=1, sticky="ew")

        ttk.Label(mainframe, text="Input Image:").grid(
            row=2, column=0, sticky="ew")

        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.grid(row=2, column=1, sticky="ew")

        for child in mainframe.winfo_children():
            child.grid_configure(padx=10, pady=5)

    def open_file(self):
        file_types = (("jpg", "*.jpg"), ("jpeg", "*.jpeg"))
        self.filename = filedialog.askopenfilename(
            title="Choose an image", initialdir=str(CURRENT), filetypes=file_types)
        img = ImageTk.PhotoImage(Image.open(self.filename))
        self.canvas.create_image(20, 20, anchor="nw", image=img)
        print(self.filename)


demo_gui(root, [])
root.mainloop()
