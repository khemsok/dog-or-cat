import os
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
model = load_model('classifier')
graph = tf.get_default_graph()
# prediction = 'yo'

APP_ROOT = os.path.dirname(os.path.abspath('__file__'))

@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():
    target = os.path.join(APP_ROOT, 'images/')
    # print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        # print(file)
        filename = 'test.jpg'
        destination = '/'.join([target, filename])
        # print(destination)
        file.save(destination)

    test_image = image.load_img('./images/test.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    with graph.as_default():
        result = model.predict(test_image)
        print(result)
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'

        print('THE RESULT IS ' + prediction)

    return render_template('index.html', prediction=prediction.upper())


app.run()
