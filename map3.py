import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load data from CSV file
df = pd.read_csv('geo_points.csv')

# Function to convert time to numerical features
def process_time(time_str):
    time_parts = time_str.split(' ')[0].split(':')
    hour, minute, second = map(int, time_parts)
    am_pm = 0 if 'AM' in time_str else 1  # 0 for AM, 1 for PM
    return hour, minute, second, am_pm

# Apply the function to the Time column
df[['Hour', 'Minute', 'Second', 'AM_PM']] = df['Time'].apply(lambda x: pd.Series(process_time(x)))

# Create cyclical features for hour, minute, and second
df['Hour_sin'] = np.sin(df['Hour'] * (2. * np.pi / 24))
df['Hour_cos'] = np.cos(df['Hour'] * (2. * np.pi / 24))
df['Minute_sin'] = np.sin(df['Minute'] * (2. * np.pi / 60))
df['Minute_cos'] = np.cos(df['Minute'] * (2. * np.pi / 60))

# Drop the original Time column and Count column (assuming it's not needed)
df = df.drop(columns=['Time', 'Count', 'Hour', 'Minute', 'Second'])

# Encode the Zone column to numerical values
label_encoder = LabelEncoder()
df['Zone'] = label_encoder.fit_transform(df['Zone'])

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and target variable from training dataset
X_train = train.drop(columns='Zone')
y_train = train['Zone']

# Separate features and target variable from testing dataset
X_test = test.drop(columns='Zone')
y_test = test['Zone']

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define different classifiers
clf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf3 = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
clf4 = SVC(kernel='linear', C=1, probability=True, random_state=42)
clf5 = KNeighborsClassifier(n_neighbors=5)

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_rf = GridSearchCV(clf1, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 4],
}

grid_gb = GridSearchCV(clf2, param_grid=param_grid_gb, cv=5, n_jobs=-1)
grid_gb.fit(X_train_scaled, y_train)
best_gb = grid_gb.best_estimator_

# Hyperparameter tuning for AdaBoost
param_grid_ada = {
    'n_estimators': [100, 200],
    'learning_rate': [1.0, 0.5],
}

grid_ada = GridSearchCV(clf3, param_grid=param_grid_ada, cv=5, n_jobs=-1)
grid_ada.fit(X_train_scaled, y_train)
best_ada = grid_ada.best_estimator_

# Hyperparameter tuning for SVM
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

grid_svc = GridSearchCV(clf4, param_grid=param_grid_svc, cv=5, n_jobs=-1)
grid_svc.fit(X_train_scaled, y_train)
best_svc = grid_svc.best_estimator_

# Hyperparameter tuning for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
}

grid_knn = GridSearchCV(clf5, param_grid=param_grid_knn, cv=5, n_jobs=-1)
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_

# Update the ensemble with the best models
eclf = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('gb', best_gb),
    ('ada', best_ada),
    ('svc', best_svc),
    ('knn', best_knn)
], voting='soft')

# Fit the ensemble model to the training data
eclf.fit(X_train_scaled, y_train)

# Make predictions on the testing data
predictions = eclf.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')