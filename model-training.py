# Import libraries
import pandas as pd
from time import time
from joblib import dump
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer

# Load data
df = pd.read_csv("data/star_data.csv")
df.columns = df.columns.str.strip()

# Define feature types
feature_types = {
    "Star_Color": "clf",
    "Spectral_Class": "clf",
    "Star_Type": "clf",
    "Temperature": "reg",
    "Radius": "reg",
    "Luminosity": "reg",
    "Absolute_Magnitude": "reg",
}

# Categorical and numeric columns (note: all inputs except the target)
categorical_features = ["Star_Color", "Spectral_Class", "Star_Type"]
numerical_features = ["Temperature", "Radius", "Luminosity", "Absolute_Magnitude"]

models = {}

for feature, ftype in feature_types.items():
    print(f"🚀  Training model to predict: {feature}")
    start_time = time()

    # Define X and y
    X = df.drop(columns=[feature])
    y = df[feature]

    # Update feature lists for this specific target
    categorical_cols = [col for col in categorical_features if col != feature]
    numerical_cols = [col for col in numerical_features if col != feature]

    # Pipeline to transform numerical columns
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pow_tnfr', PowerTransformer(method='yeo-johnson', standardize=True))
    ])

    # Pipeline to transform categorical columns
    categoric_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer to encode categorical columns + scale/transform numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categoric_pipeline, categorical_cols),
            ("num", numeric_pipeline, numerical_cols)
        ]
    )

    # Define model type
    model = RandomForestClassifier() if ftype == "clf" else RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)

    # Create pipeline
    master_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit model
    master_pipeline.fit(X_train, y_train)

    # Save model
    models[feature] = master_pipeline
    dump((master_pipeline, X_train.columns.tolist()), f"models/{feature.lower().replace('_', '-')}-predictor.joblib", compress=3)

    # Evaluate
    y_pred = master_pipeline.predict(X_test)
    if ftype == "clf":
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"✅  Accuracy Score: {acc:.4f}%")
    else:
        r2 = r2_score(y_test, y_pred) * 100
        print(f"✅  R2 Score: {r2:.4f}%")

    print(f"⏱️  Training completed in {time() - start_time:.2f} seconds\n")

print("🎉 All MODELS trained and saved!")
