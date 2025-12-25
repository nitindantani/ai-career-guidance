import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
FEATURES = [
    "stream",
    "subject_liked",
    "skills",
    "soft_skill",
    "preferred_field"
]
TARGET = "career_label"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")

# Normalize text
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)

X = df[FEATURES]
y = df[TARGET]

# ---------------- PREPROCESSOR ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)
    ]
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# ---------------- PIPELINE ----------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# ---------------- SAVE ----------------
joblib.dump(pipeline, "career_model.pkl")
print("Model saved successfully")
