from flask import Flask

from iitprediction.predict.routes import main
from iitprediction.train.routes import trainer


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    app.register_blueprint(trainer)
    return app