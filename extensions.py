# extensions.py
from flask_sqlalchemy import SQLAlchemy
from queue import Queue

db = SQLAlchemy()
status_queue = Queue()