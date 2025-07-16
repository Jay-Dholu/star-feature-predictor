from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Prediction
from forms import SignUp, Login, Input
from datetime import datetime
from decimal import Decimal
from pytz import timezone
import pandas as pd
import joblib
import json
import os


IST = timezone('Asia/Kolkata')
CATEGORICAL_COLUMNS = ["star_color", "spectral_class", "star_type"]
STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}
FEATURE_COLUMNS = [
    "star_color",
    "spectral_class",
    "star_type",
    "temperature",
    "radius",
    "luminosity",
    "absolute_magnitude",
]
UNIQUE_VALUES = {
    "star_color": ["Blue", "White", "Yellow", "Orange", "Red"],
    "spectral_class": ["O", "B", "A", "F", "G", "K", "M"],
    "star_type": ["Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant"],
}
RANGES = {
    "temperature": "Range [300, 50000]",
    "radius": "Range (0, 1000]",
    "luminosity": "Range [0, 1.0e+7]",
    "absolute_magnitude": "Range [-12, 25]"
}
FEATURE_MAP = {
    "star_color": "Star_Color",
    "spectral_class": "Spectral_Class",
    "star_type": "Star_Type",
    "temperature": "Temperature",
    "radius": "Radius",
    "luminosity": "Luminosity",
    "absolute_magnitude": "Absolute_Magnitude"
}


app = Flask(__name__)
app.config['SECRET_KEY'] = "SmF5IGlzIGtpbmcgb2YgdGhlIHVuaXZlcnNlLiBIZSBpcyB1bmRlZmVhdGFibGUsIGV2ZW4gYnkgQWxpZW5zLg=="
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db.init_app(app)
with app.app_context():
    db.create_all()


def load_models():
    model_dir = "models"
    filenames = {
        "absolute-magnitude": "absolute-magnitude-predictor.joblib",
        "luminosity": "luminosity-predictor.joblib",
        "radius": "radius-predictor.joblib",
        "spectral-class": "spectral-class-predictor.joblib",
        "star-color": "star-color-predictor.joblib",
        "star-type": "star-type-predictor.joblib",
        "temperature": "temperature-predictor.joblib",
    }

    models = {}

    for key, filename in filenames.items():
        full_path = os.path.join(model_dir, filename)
        try:
            model_tuple = joblib.load(full_path)
            if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
                models[key] = {
                    "model": model_tuple[0],
                    "features": model_tuple[1]
                }
            else:
                raise ValueError("Model file format invalid (expected tuple)")
        except Exception as e:
            print(f"[ERROR] Failed to load model for '{key}': {e}")
            models[key] = None

    return models


def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


MODELS = load_models()


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title="Star Feature Predictor")


@app.route("/sign_up", methods=['GET', 'POST'])
def sign_up():
    form = SignUp()
    if form.validate_on_submit():
        name = form.name.data
        username = form.username.data
        email = form.email.data
        password = form.password.data

        existing_username = User.query.filter(User.username == username).first()
        existing_email = User.query.filter(User.email == email).first()

        if existing_username and existing_email:
            flash("Username and Email already exists!", "danger")
            return redirect(url_for('sign_up'))

        if existing_username:
            flash("Username already exists!", "danger")
            return redirect(url_for('signup'))

        if existing_email:
            flash("Email already exists!", "danger")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        new_user = User(name=name, username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('sign_up.html', title="Sign Up | Star Feature Predictor", form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = Login()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = User.query.filter_by(email=email).first()
        next_page = request.args.get('next')

        email_exits = User.query.filter(User.email == email).first()

        if not email_exits:
            flash("Email doesn't exist! You can create account with this email!", "info")
            return redirect(url_for('login'))

        if user and check_password_hash(user.password, password):
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Logged in successfully!", "success")
            return redirect(next_page) if next_page else redirect(url_for('prediction_options'))
        else:
            flash("Invalid credentials. Try again.", "danger")
            return redirect(url_for('login'))
        
    return render_template('login.html', title="Login | Star Feature Predictor", form=form)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logout successfully!", "success")
    return redirect(url_for('home'))


@app.route("/about")
def about():
    return render_template('about.html', title="About  | Star Feature Predictor")


@app.route("/prediction_options")
def prediction_options():
    return render_template('prediction_options.html', title="Prediction Options  | Star Feature Predictor")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash("You need to log in first to make predictions.", "warning")
        return redirect(url_for('login', next=request.full_path))

    target = request.args.get('target')
    form = Input()

    if not target:
        flash("Missing 'feature' parameter in the URL.", "danger")
        return redirect(url_for('prediction_options'))

    try:
        model_entry = MODELS.get(target.replace("_", "-").lower())

        if not model_entry:
            flash("Model not found or failed to load.", "danger")
            return redirect(url_for("prediction_options"))

        model = model_entry["model"]
        expected_features = model_entry["features"]

    except Exception as e:
        print(f"Model loading error: {e}")
        flash("Model file not found or unreadable.", "danger")
        return redirect(url_for('prediction_options'))

    if target in form._fields:  # Remove field being predicted
        del form._fields[target]

    if form.validate_on_submit():
        try:
            input_data = {
                "Star_Color": form.star_color.data,
                "Star_Type": form.star_type.data,
                "Spectral_Class": form.spectral_class.data,
                "Temperature": form.temperature.data,
                "Radius": form.radius.data,
                "Luminosity": form.luminosity.data,
                "Absolute_Magnitude": form.absolute_magnitude.data,
            }

            input_data.pop(FEATURE_MAP[target], None)
            input_df = pd.DataFrame([input_data])[expected_features]
            prediction = model.predict(input_df)[0]

            if isinstance(prediction, Decimal):
                prediction = float(prediction)
            
            # Safely assign predicted label if star_type was target
            if "Star_Type" in input_data:
                input_data["Star_Type"] = STAR_TYPES.get(int(input_data["Star_Type"]), prediction)
            elif target == "star_type":
                input_data["Star_Type"] = STAR_TYPES.get(prediction, prediction)

            new_prediction = Prediction(
                prediction_time=datetime.now(IST),
                user_id=session['user_id'],
                input_data=json.dumps(input_data, default=convert_decimal),
                predicted_feature=target.replace("_", " ").title(),
                prediction = prediction if target != "star_type" else STAR_TYPES.get(prediction, prediction)
            )
            db.session.add(new_prediction)
            db.session.commit()

            return redirect(url_for('past_predictions', target=target, model_prediction=prediction))

        except Exception as e:
            print(f"Error during prediction:\n{e}")
            flash("Something went wrong during prediction. Please try again.", "danger")
            return redirect(url_for('predict', target=target))

    return render_template(
        "predict.html",
        title="Predict | Star Feature Predictor",
        columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        unique_values=UNIQUE_VALUES,
        ranges=RANGES,
        target=target,
        form=form
    )


@app.route("/history")
@app.route("/past_predictions")
def past_predictions():
    if 'user_id' not in session:
        flash("Please log in to view your past predictions.", "warning")
        return redirect(url_for('login'))

    target = request.args.get('target')
    model_prediction = request.args.get('model_prediction')
    flash("You can see your past predictions here.", "info")

    user_predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.prediction_time.desc()).all()

    for prediction in user_predictions:
        try:
            if isinstance(prediction.input_data, str):
                prediction.input_data = json.loads(prediction.input_data.replace("'", '"'))
            elif isinstance(prediction.input_data, dict):
                prediction.input_data = json.dumps(prediction.input_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for prediction {prediction.id}: {e}")
            flash(f"Error decoding data for prediction {prediction.id}.", "danger")
            continue

    return render_template('past_predictions.html', title="Past Predictions | Star Feature Predictor", predictions=user_predictions, target=target, model_prediction=model_prediction)


@app.route("/delete")
@app.route("/delete_prediction/<int:prediction_id>", methods=["GET", "POST"])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        flash("Please log in to delete your predictions.", "warning")
        return redirect(url_for('login'))

    prediction = Prediction.query.filter_by(id=prediction_id, user_id=session['user_id']).first()

    if prediction:
        db.session.delete(prediction)
        db.session.commit()
        flash("Prediction has been deleted.", "success")
    else:
        flash("Prediction not found or you don't have permission to delete it.", "danger")

    return redirect(url_for('past_predictions'))


if __name__ == '__main__':
    app.run(debug=True)
    