# app.py
from flask import Flask, app
from extensions import db
from routes.main_routes import main_bp
from routes.admin_routes import admin_bp
from utils import load_encodings_from_db, fix_attendance_user_ids
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
import click
from config import SECRET_KEY

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.secret_key = SECRET_KEY
    db.init_app(app)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

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
        fix_attendance_user_ids()
    app.run(debug=True, threaded=True)