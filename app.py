from flask import Flask, render_template, request

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# creating a dictionary for identifying the vehicle classes
dic = {0: 'Ambulance',
       1: 'Bus',
       2: 'Car',
       3: 'Limousine',
       4: 'Motorcycle',
       5: 'Taxi',
       6: 'Truck',
       7: 'Van'}

# loading the model
model_path = os.listdir("../assets/vehicle-detection-72percent.tflite")
model = load_model(model_path)

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    p = model.predict_classes(i)
    return dic[p[0]]


@app.route('/', methods=['GET'])
def helloWorld():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    if request.method == 'POST':
        predict_img = request.files['predict_image']
        save_path = "./images/" + predict_img.filename
        predict_img.save(save_path)

        p = predict_label(save_path)

    return render_template('index.html', prediction=p, img_path=save_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
