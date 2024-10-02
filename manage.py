import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import pickle

# Load your dataset
# Use forward slashes or raw strings for file paths to avoid escape character issues
data = pd.read_csv('.venv/parkinsons.data')  
data.drop(['name'], axis=1, inplace=True)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Define features and target
X = data.drop(['status'], axis=1)
y = data['status']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize and train the XGBoost Classifier
xgb_xlf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_xlf.fit(x_train_scaled, y_train)

# Get Feature Importances
importances = xgb_xlf.feature_importances_
feature_names = X.columns

# Create a Series and sort
feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title('Feature Importances from XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Select top 10 features
k = 10
top_features = feat_importances.index[:k]
print("Top Features:", top_features.tolist())

# Reduce the dataset to top 10 features + target
X_top = data[top_features.tolist()]
y_top = data['status']

# Split the top features data
x_train_top, x_test_top, y_train_top, y_test_top = train_test_split(
    X_top, y_top, test_size=0.2, random_state=42, stratify=y_top
)

# Scale the top features
scaler_top = StandardScaler()
x_train_top_scaled = scaler_top.fit_transform(x_train_top)
x_test_top_scaled = scaler_top.transform(x_test_top)

# Train XGBoost on top features
xgb_top = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_top.fit(x_train_top_scaled, y_train_top)

# Make predictions
y_pred = xgb_top.predict(x_test_top_scaled)
y_pred_prob = xgb_top.predict_proba(x_test_top_scaled)[:, 1]

# Evaluate the model
cm = confusion_matrix(y_test_top, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print("Classification Report:\n", classification_report(y_test_top, y_pred))

roc_auc = roc_auc_score(y_test_top, y_pred_prob)
print("ROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_top, y_pred_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Define label mapping
label_mapping = {0: 'Healthy', 1: 'Parkinson\'s Disease'}

# Map predictions to labels
y_pred_labels = [label_mapping[label] for label in y_pred]



# Create a DataFrame for comparison
results = pd.DataFrame({
    'Actual': [label_mapping[label] for label in y_test_top],
    'Predicted': y_pred_labels,
    'Probability (%)': [round(prob * 100, 2) for prob in y_pred_prob]
})

print(results.head())

# Serialize the model and scaler
with open('xgboost_model_top.pkl', 'wb') as f:
    pickle.dump(xgb_top, f)

with open('scaler_top.pkl', 'wb') as f:
    pickle.dump(scaler_top, f)
