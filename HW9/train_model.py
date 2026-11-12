# train_model.py — Week 12: produces model.joblib (the model the server wraps)
# Run once:  python train_model.py
# Regenerate whenever scikit-learn versions differ (pickles are version-sensitive!).
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f"Held-out accuracy: {model.score(X_te, y_te):.3f}")

joblib.dump(
    {"model": model,
     "feature_names": list(X.columns),
     "class_names": list(iris.target_names)},
    "model.joblib",
)
print("Saved model.joblib")
