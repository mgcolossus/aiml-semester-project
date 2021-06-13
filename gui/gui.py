from tkinter import *
from tkinter import filedialog, ttk
from tkinter import filedialog as fd
from PIL import Image
from PIL import ImageTk, ImageOps

from tensorflow import keras
import tensorflow as tf
from keras import backend as K
import re
import numpy as np
import shutil
import os
import json
import cv2


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def read_image(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()

    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def pairs_for_one_image(path_to_photo):
    path, dirs, files = next(os.walk("archive"))
    dirs_count = len(dirs)

    image = Image.open(path_to_photo)
    image = np.array(image.resize((92, 112)))
    image = image[::2, ::2]
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    x_pair = np.zeros([dirs_count, 2, 1, dim1, dim2])
    for i in range(len(dirs)):
        img_p = read_image('archive/s' + str(i + 1) + '/' + str(np.random.randint(9) + 1) + '.pgm', 'rw+')
        img_p = img_p[::2, ::2]
        x_pair[i, 0, 0, :, :] = image
        x_pair[i, 1, 0, :, :] = img_p
    return x_pair / 255

def predict_one_image(my_model, path_to_photo):
    x_final_test = pairs_for_one_image(path_to_photo)
    final_pred = my_model.predict([x_final_test[:, 0], x_final_test[:, 1]])
    # final_pred
    best_match = final_pred.argmin() + 1
    if final_pred[best_match - 1] < 0.5:
        with open("archive/names.json") as f:
            data = json.load(f)
            return data[str(best_match)]
    else:
        return "Undefined"


model = keras.models.load_model("model", custom_objects={"contrastive_loss": contrastive_loss})
model.summary()

prototxt = 'deploy.prototxt'
model_cv = 'res10_300x300_ssd_iter_140000.caffemodel'
model_cv = cv2.dnn.readNetFromCaffe(prototxt, model_cv)


def check_photo(my_model, path_to_photo, person_name):
    image = cv2.imread(path_to_photo)
    photo_extension = path_to_photo.split(".")[1]
    if (photo_extension == "pgm"):
        path_to_result_photo = path_to_photo.split(".")[0] + "_result.jpg"
        if os.path.isfile(path_to_result_photo):
            os.remove(path_to_result_photo)
        cv2.imwrite(path_to_result_photo, image)
        os.rename(path_to_result_photo, path_to_result_photo.split(".")[0] + ".pgm")
        path_to_result_photo = path_to_result_photo.split(".")[0] + ".pgm"
    else:
        path_to_result_photo = path_to_photo.split(".")[0] + "_result." + path_to_photo.split(".")[1]
        if os.path.isfile(path_to_result_photo):
            os.remove(path_to_result_photo)
        cv2.imwrite(path_to_result_photo, image)


    if not os.path.isdir('frames'):
        os.mkdir("frames")
    else:
        shutil.rmtree('frames')

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model_cv.setInput(blob)
    detections = model_cv.forward()

    path, dirs, files = next(os.walk("frames"))
    os.mkdir("frames/z" + str(len(dirs) + 1))

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]

            path, dirs1, files = next(os.walk("frames"))
            cv2.imwrite("frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".jpg", frame)
            os.rename(os.path.join("frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".jpg"),
                      os.path.join("frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".pgm"))

            img = Image.open("frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".pgm")
            img = ImageOps.grayscale(img)
            img.save("frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".pgm")
            pr = predict_one_image(my_model, "frames/z" + str(len(dirs) + 1) + "/" + str(len(files) + 1) + ".pgm")
            if pr == person_name:
                image = cv2.imread(path_to_result_photo)
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
                cv2.putText(image, pr, (startX, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                if (photo_extension == "pgm"):
                    path_to_result_photo = path_to_result_photo.split(".")[0] + ".jpg"
                    cv2.imwrite(path_to_result_photo, image)
                    os.rename(path_to_result_photo, path_to_result_photo.split(".")[0] + ".pgm")
                else:
                    cv2.imwrite(path_to_result_photo, image)

    if os.path.isdir('frames'):
        shutil.rmtree('frames')

    return path_to_result_photo


class App:
    def __init__(self, window=Tk()):
        self.window = window
        self.window.title("GUI")
        print(self.window.winfo_screenheight())
        print(self.window.winfo_screenwidth())
        self.image_size = [600, 400]
        self.window.geometry(str(self.window.winfo_screenwidth()-300)+"x"+str(self.window.winfo_screenheight()-300))
        self.names = [
            "Barak Obama",
            "James",
            "Robert",
            "John",
            "Michael",
            "William",
            "David",
            "Richard",
            "Joseph",
            "Thomas",
            "Charles",
            "Christopher",
            "Daniel",
            "Matthew",
            "Anthony",
            "Mark",
            "Donald",
            "Steven",
            "Paul",
            "Andrew",
            "Joshua",
            "Kenneth",
            "Kevin",
            "Brian",
            "George",
            "Edward",
            "Ronald",
            "Timothy",
            "Jason",
            "Jeffrey",
            "Ryan",
            "Jacob",
            "Gary",
            "Nicholas",
            "Eric",
            "Jonathan",
            "Stephen",
            "Larry",
            "Justin",
            "Scott",
            "Brandon"]
        self.filepath = ""
        self.canvas = Canvas(self.window, width=600, height=400)
        self.init_load_image()
        self.image_label = Label(self.canvas, image=self.fig_image)
        self.image_label.pack()

        self.open_file_button = Button(window, text="Открыть фотографию", command=self.update_image)
        self.open_file_button.pack()

        self.combo_label = Label(text="Выберите человека для распознавания")
        self.combo_label.pack()
        self.combo = ttk.Combobox(window, values=tuple(self.names))
        self.combo.pack()
        self.combo.current(0)
        self.open_file_button = Button(window, text="Распознать", command=self.recognize_click)
        self.open_file_button.pack()

        self.canvas.pack()

    def recognize_click(self):
        selected_name = self.names[self.combo.current()]
        result_path = check_photo(model, self.filepath, selected_name)
        self.fig_image = ImageTk.PhotoImage(Image.open(result_path).resize(self.image_size))
        self.image_label.config(image=self.fig_image)

    def init_load_image(self):
        self.fig_image = ImageTk.PhotoImage(Image.open("empty.jpg"))

    def load_image(self):
        self.filepath = filedialog.askopenfilename()
        self.fig_image = ImageTk.PhotoImage(Image.open(self.filepath).resize(self.image_size))

    def update_image(self):
        self.load_image()
        self.image_label.config(image=self.fig_image)

    def mainloop(self):
        self.window.mainloop()
        print('mainloop closed...')


if __name__ == '__main__':
    import time
    app = App()
    app.mainloop()
