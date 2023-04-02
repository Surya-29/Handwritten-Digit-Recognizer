from flask import Flask, render_template, request, redirect, url_for, session, flash
import torch
import os
import mysql.connector
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import json

import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")


app = Flask(__name__)
app.config[
    "SECRET_KEY"
] = "1b11e3688e45e9f809a8e11f5fc3fdfe1041a8a808da7f0a1192fc3b778055281ef5d073c3a36ddfe193"


dataBase = mysql.connector.connect(
    host="localhost", user="root", passwd="_______", database="userdata"
)
cursorObject = dataBase.cursor()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)


@app.route("/createuser", methods=["POST"])
def createUser():
    if request.method == "POST":
        username = request.form.get("uname")
        pwd = request.form.get("pwd")
        cpwd = request.form.get("cpwd")
        if cpwd == pwd:
            val = (username, pwd)
            cursorObject.execute(
                "INSERT INTO users (name, passwd) VALUES (%s, %s)", val
            )
            dataBase.commit()

            if databaseConnection(username, pwd):
                flash("Account successfully created!")
                return render_template("login.html", signup=False)
        else:
            return render_template("login.html", signup=True, messages=True)


def databaseConnection(username, password):
    # cursorObject.execute("CREATE DATABASE IF EXISTS userdata")
    # cursorObject.execute("SHOW DATABASES")
    # cursorObject.execute("CREATE TABLE users (name VARCHAR(255), passwd VARCHAR(255))")

    cursorObject.execute("SELECT * from users")
    datalist = cursorObject.fetchall()

    print(datalist)
    # print((username, password))
    if (username, password) in datalist:
        return True
    else:
        False


cnn_model = Net()
label = 0
accuracy = 0


def pred(image):
    global label, accuracy
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR) / 255.0
    # resized_image = resized_image.astype("float64")
    data = torch.from_numpy(resized_image)
    data = data.type("torch.FloatTensor")
    data = data.unsqueeze(0)

    with open("model.pkl", "rb") as f:
        cnn_model = pickle.load(f)

    with torch.no_grad():
        output = cnn_model(data)

    pred = np.round_(np.array(output), decimals=3) * 100
    label = int(np.argmax(pred))
    accuracy = int(pred[0][label])
    print((label, accuracy))
    return (label, accuracy)


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("login.html")


@app.route("/loginuser", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        username = request.form.get("uname")
        pwd = request.form.get("pwd")

        if databaseConnection(username, pwd):
            return redirect(url_for("classifierPage"))
        else:
            return render_template("login.html", signup=True)
    if request.method == "GET":
        print("GET")
    return render_template("login.html", signup=False)


@app.route("/NoClassifier")
def classifierPage(model_pred=(True, None, None)):
    return render_template("classifier.html")


@app.route("/getImage", methods=["POST", "GET"])
def getImage():
    global label, accuracy

    if request.method == "POST":

        imagebase64 = request.form["imagebase64"]
        encoded_img = imagebase64.split(",")[1]
        image_data = bytes(encoded_img, encoding="ascii")
        img_conv = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")
        img_conv.save("image.png")
        image = np.asarray(img_conv)
        return json.dumps({"predictions": pred(image)})

    else:

        return json.dumps({"predictions": (label, accuracy)})


if __name__ == "__main__":
    app.run(debug=True)
