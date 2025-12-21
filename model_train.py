import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------------------------------
# CONFIG (FIXED FEATURE ORDER)
# --------------------------------------------------
FEATURES = [
    "stream",
    "subject_liked",
    "skills",
    "soft_skill",
    "preferred_field"
]
TARGET = "career_label"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("career_data.csv")

# Normalize strings
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)

# --------------------------------------------------
# VALIDATION (IMPORTANT)
# --------------------------------------------------
missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# --------------------------------------------------
# ENCODE FEATURES
# --------------------------------------------------
encoders = {}

for col in FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------
target_encoder = LabelEncoder()
df[TARGET] = target_encoder.fit_transform(df[TARGET])

# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# SAVE ARTIFACTS
# --------------------------------------------------
joblib.dump(model, "career_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model and encoders saved successfully")
