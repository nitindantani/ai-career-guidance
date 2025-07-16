import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data
df = pd.read_csv("career_data.csv")

# Separate the target column
target_col = "career_label"
feature_cols = [col for col in df.columns if col != target_col]

# Encode features
encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target separately
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])

# Prepare training data
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "career_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")