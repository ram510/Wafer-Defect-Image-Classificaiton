import os
import numpy as np

from flask import request, render_template
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app = Flask(__name__)
UPLOAD_FOLDER = "C://Users//rchilakamarr//PycharmProjects//WaferDefectClassification//Static"


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        image_file = request.files["img"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            pred = predict(image_location)
            # print(pred)
            return render_template("index.html", classification=pred, image_loc=image_file.filename)

    return render_template("index.html", classification='Select an image', image_loc=None)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
saved_model = load_model('C://Users//rchilakamarr//PycharmProjects//WaferDefectClassification//model.h5')
class_indices = {'Center': 0,
                 'Donut': 1,
                 'Edge-loc': 2,
                 'Edge-ring': 3,
                 'Loc': 4,
                 'Near-Full': 5,
                 'None': 6,
                 'Random': 7,
                 'Scratch': 8}


def predict(image_filepath):
    img_shape = (64, 65, 4)
    eval_image = image.load_img(image_filepath, target_size=img_shape, color_mode='rgba')
    eval_image = image.img_to_array(eval_image)
    eval_image = np.expand_dims(eval_image, axis=0)
    pred_list = saved_model.predict(eval_image)
    keys = list(class_indices.keys())
    return 'wafer defect classified as ' + str(keys[pred_list.argmax()])
