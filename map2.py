import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

# Create the models
clf1 = RandomForestClassifier(random_state=42)
clf2 = SVC(probability=True, random_state=42)
clf3 = KNeighborsClassifier()

# Create an ensemble of the models
eclf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('knn', clf3)], voting='soft')

# Define hyperparameters to tune
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'svc__C': [0.1, 1, 10],
    'knn__n_neighbors': [3, 5, 7]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(eclf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the testing data
predictions = best_model.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

