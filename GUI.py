import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pathlib
from tensorflow.io import read_file, decode_jpeg
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
CURRENT = pathlib.Path(__file__).parent

# Load models
# flickr_300K = load_model("flickr.h5")
# flickr_300K.compile()
# models = {
#     "Flickr 25K Dataset GAN (300K Steps):": flickr_300K,
# }


def load(image_file):
    """Read and decode an image file to a uint8 tensor"""
    image = read_file(image_file)
    image = decode_jpeg(image)

    return tf.cast(image, tf.float32)


def resize(image, height, width):
    image = tf.image.resize(image,
                            [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image


def normalize(image):
    return (image / 127.5) - 1


def load_image(image_file):
    target = load(image_file)
    target = resize(target, 384, 384)
    target = normalize(target)

    bw = tf.image.rgb_to_grayscale(target)
    bw = tf.concat([bw, bw, bw], axis=2)
    return bw, target


def load_test_image(image_file):
    target = load(image_file)
    target = resize(target, 384, 384)

    bw = target
    if bw.shape[-1] == 3:
        bw = tf.image.rgb_to_grayscale(bw)

    bw = tf.concat([bw, bw, bw], axis=2)

    target = normalize(target)
    bw = normalize(bw)

    return bw, target


class Demo_GUI():
    def __init__(self, root, models):
        # models should be a dictionary
        # name of the model: loaded model object
        self.models = models

        root.title("Image Colorization Demo")

        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky="eswn")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.filename = tk.StringVar()
        self.pick_button = ttk.Button(
            root, text="Choose an Image File", command=self.process).grid(row=2, column=0, sticky="ew")

        ttk.Label(self.mainframe, textvariable=self.filename).grid(
            row=1, column=1, sticky="ew")
        ttk.Label(self.mainframe, text="Input File:").grid(
            row=1, column=0, sticky="ew")
        ttk.Label(self.mainframe, text="Input Image:").grid(
            row=3, column=0, sticky="ew")

        self.input_image = None
        self.target = None

        for i, name in enumerate(self.models):
            ttk.Label(self.mainframe, text=name).grid(
                row=4+i, column=0, sticky="ew")

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=10, pady=5)

    def process(self):
        file_types = (("jpg", "*.jpg"), ("jpeg", "*.jpeg"),
                      ("JPG", "*.JPG"), ("JPEG", "*.JPEG"))
        self.filename.set(filedialog.askopenfilename(
            title="Choose an image", initialdir=str(CURRENT), filetypes=file_types))
        self.input_image, self.target = load_test_image(self.filename.get())
        self.show_image(self.target, 3, 1)
        self.make_predictions()

    def show_image(self, img, row, col):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(img * 0.5 + 0.5)
        plt.axis("off")
        canvas = FigureCanvasTkAgg(fig, master=self.mainframe)
        canvas.get_tk_widget().grid(row=row, column=col, sticky="ew")
        canvas.get_tk_widget().grid_configure(padx=10, pady=5)
        canvas.draw()

    def make_predictions(self):
        GAN_input = tf.expand_dims(self.input_image, axis=0)

        for i, model in enumerate(self.models.values()):
            self.show_image(model(GAN_input, training=False)[0], 4+i, 1)


def main():
    Demo_GUI(root, models)
    root.mainloop()


if __name__ == "__main__":
    main()
