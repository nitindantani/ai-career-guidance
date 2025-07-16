import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load and encode
df = pd.read_csv("career_data.csv")
le_dict = {col: LabelEncoder().fit(df[col]) for col in df.columns}
for col, le in le_dict.items():
    df[col] = le.transform(df[col])

# Train model
X = df.drop("career_label", axis=1)
y = df["career_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

# Save
joblib.dump(model, "career_model.pkl")
joblib.dump(le_dict, "encoders.pkl")
