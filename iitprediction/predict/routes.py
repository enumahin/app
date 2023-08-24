from flask import Blueprint, render_template, request
import logging
import getpass

import iitprediction.predict.predict

LOG_LEVEL = logging.INFO
# Set a seed to enable reproducibility
SEED = 1

# Get the username of the person who is running the script.
USERNAME = getpass.getuser()

# Set a format to the logs.
LOG_FORMAT = '[%(levelname)s | ' + USERNAME + ' | %(asctime)s] - %(message)s'

LOG_FILENAME = "training.log"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"),
              logging.StreamHandler()]
)
main = Blueprint("main", __name__)


@main.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        total = []
        linelist = request.files['linelist']
        status = iitprediction.predict.predict.predict_status(linelist)
        if status[0] == 'InActive':
            total = iitprediction.predict.predict.predict_number_of_visits(linelist)
        response = {"status": status[0], "total": total[0]}
        print(response)
        return response
    else:
        # logging.info("############################## WELCOME ##########################")
        return render_template("predict/index.html")