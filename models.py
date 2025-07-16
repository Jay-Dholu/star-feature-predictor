from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone


ist = timezone('Asia/Kolkata')
db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(30), nullable=False, unique=True)
    email = db.Column(db.String(30), nullable=False, unique=True)
    password = db.Column(db.String(30), nullable=False)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_time = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(ist))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    predicted_feature = db.Column(db.String(20), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
