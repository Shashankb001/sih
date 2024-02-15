import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

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

# Create an ensemble of the models
eclf = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('ada', clf3)], voting='soft')

# Fit the ensemble model to the training data
eclf.fit(X_train_scaled, y_train)

# Make predictions on the testing data
predictions = eclf.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
