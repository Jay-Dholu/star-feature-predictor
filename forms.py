from wtforms import StringField, SelectField, PasswordField, DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Email, Length, EqualTo
from flask_wtf import FlaskForm
from decimal import Decimal
import pandas as pd


data = pd.read_csv(r'data\star_data.csv')
STAR_TYPES = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}


class SignUp(FlaskForm):
    name = StringField(label="Name", validators=[DataRequired()])
    username = StringField(label="Unique User Name", validators=[DataRequired(), Length(2, 30)])
    email = StringField(label="Email", validators=[DataRequired(), Email()])
    password = PasswordField(label="Password", validators=[DataRequired(), Length(8, 30)])
    confirm_password = PasswordField(label="Confirm Password", validators=[DataRequired(), Length(8, 30), EqualTo('password')])
    submit = SubmitField("Sign Up")


class Login(FlaskForm):
    email = StringField(label="Email", validators=[DataRequired(), Email()])
    password = PasswordField(label="Password", validators=[DataRequired(), Length(8, 30)])
    submit = SubmitField("Login")


class Input(FlaskForm):
    radius = DecimalField(
        label='Radius (in R☉)',
        places=16,
        rounding=None,
        validators=[
            DataRequired(),
            NumberRange(
                min=Decimal('0.01').quantize(Decimal('1.0000000000000000')),
                max=Decimal('1000').quantize(Decimal('1.0000000000000000')),
                message='Value must be between (0, 1000]  with decimal precision upto 16 digits.'
            )
        ]
    )
    luminosity = DecimalField(
        label='Luminosity (in L☉)',
        places=16,
        rounding=None,
        validators=[
            DataRequired(),
            NumberRange(
                min=Decimal('0').quantize(Decimal('1.0000000000000000')),
                max=Decimal('1.0e+07').quantize(Decimal('1.0000000000000000')),
                message='Value must be between [0, 1.0e+07] with decimal precision upto 16 digits.'
            )
        ]
    )
    temperature = DecimalField(
        label='Temperature (in K)',
        places=16,
        rounding=None,
        validators=[
            DataRequired(),
            NumberRange(
                min=Decimal('300').quantize(Decimal('1.0000000000000000')),
                max=Decimal('50000').quantize(Decimal('1.0000000000000000')),
                message='Value must be between [300, 50000] with decimal precision upto 16 digits.'
            )
        ]
    )
    absolute_magnitude = DecimalField(
        label='Absolute Magnitude',
        places=16,
        rounding=None,
        validators=[
            DataRequired(),
            NumberRange(
                min=Decimal('-12').quantize(Decimal('1.0000000000000000')),
                max=Decimal('25').quantize(Decimal('1.0000000000000000')),
                message='Value must be between [-12, 25] with decimal precision upto 16 digits.'
            )
        ]
    )
    star_color = SelectField(
        label='Star Color',
        choices=data.Star_Color.unique().tolist(),
        validators=[DataRequired()]
    )
    spectral_class = SelectField(
        label='Spectral Class',
        choices=data.Spectral_Class.unique().tolist(),
        validators=[DataRequired()]
    )
    star_type = SelectField(
        label='Star Type',
        choices=[(str(k), v) for k, v in STAR_TYPES.items()],
        validators=[DataRequired()]
    )
    submit = SubmitField(
        label='Predict'
    )
