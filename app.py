from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_celery import make_celery
from celery import Celery
import os, zipfile
import sqlite3
import datetime
import Capstone_V3

DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(
        app.import_name,
    
        # backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
celery.conf.update(app.config)
# celery = make_celery(app)


# sanity check route
@app.route('/')
def index():
    return render_template('index.html')


app.config['UPLOAD_FOLDER'] = "/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images"

#Upload files to backend
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():

    msg=""
    if request.method == 'POST':

        if request.files:
            file = request.files["image"]
            try:
                name = request.form["name"]
            except:
                name = "default name"
                msg = "error in getting name"
                return render_template("index.html",msg = msg)
            try:
                fileName = (str(file.filename))
                fileName = fileName.split(".", maxsplit=1)[1]
                fileName = name + "." + fileName
                file.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], fileName))
                msg=fileName+" has been uploaded"
                name_of_file = "/Users/abhishek.venkatesh/Desktop/Capstone/Data/model_images"+fileName
                # parking_lot = Capstone_V3.Parking_Lot(name_of_file)
                print("Instance of class created")
                # parking_lot.execute_lots()
                # execute.delay(parking_lot)
                # execute.apply_async(parking_lot, expires=60)
                print("Execute is called")
                return render_template("index.html",msg = msg)
            except:
                msg = "Error in database insertion"
                return render_template("index.html",msg = msg)
        else:
            print("POST True, file not saved.")
            msg="request.files return FALSE"
    else:
        print("File not saved")
    return render_template('index.html',msg=msg)
   

@celery.task(name='app.execute')
def execute(parking_lot):
    parking_lot.execute_lots()
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
