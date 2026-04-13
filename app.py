# app.py
from flask import Flask
from extensions import db
from routes import main_bp
from face_utils import load_encodings_from_db
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
import click

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

    db.init_app(app)
    app.register_blueprint(main_bp)

    @app.cli.command("init-db")
    def init_db_command():
        with app.app_context():
            db.create_all()
            print("Database telah diinisialisasi.")

    return app

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        db.create_all()
        load_encodings_from_db()
    app.run(debug=True, threaded=True)