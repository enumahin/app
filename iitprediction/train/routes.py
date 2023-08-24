import logging
import os
import shutil
from time import sleep
from flask import Blueprint, render_template, current_app, request, redirect
from iitprediction.train.model_training import train as train_model

trainer = Blueprint("trainer", __name__)

# getting the current working directory
src_dir = os.getcwd()
des_dir = src_dir + "/iitprediction/static"

@trainer.route("/support", methods=['POST', 'GET'])
def train():
    logging.info(src_dir)
    logging.info(des_dir)
    if request.method == 'POST':
        # try:
        #     with open('training.log', 'w'):
        #         pass
        # except FileNotFoundError:
        #     pass
        linelist = request.files['linelist']
        logging.info(linelist)
        train_model(linelist)
        # importing the modules
        # copying the files
        try:
            shutil.copyfile('iit.png', des_dir + "/iit.png")
            shutil.copyfile('total_visit.png', des_dir + '/total_visit.png')
        except FileNotFoundError:
            pass
        return "GOOD"
        # return redirect(request.url)
    return render_template("train/support.html")
#
# @trainer.route("train", methods=['GET', 'POST'])
# def train_model():
#

@trainer.route("/stream")
def stream_log():
    def generate_log():
        with open('training.log', mode='w+') as file:
            while True:
                yield file.read()
                sleep(1)
    return current_app.response_class(generate_log(), mimetype='text/plain')