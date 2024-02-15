# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # Load data from CSV file
# df = pd.read_csv('geo_points.csv')

# # Function to convert time to numerical features
# def process_time(time_str):
#     time_parts = time_str.split(' ')[0].split(':')
#     hour, minute, second = map(int, time_parts)
#     am_pm = 0 if 'AM' in time_str else 1  # 0 for AM, 1 for PM
#     return hour, minute, second, am_pm

# # Apply the function to the Time column
# df[['Hour', 'Minute', 'Second', 'AM_PM']] = df['Time'].apply(lambda x: pd.Series(process_time(x)))

# # Drop the original Time column and Count column (assuming it's not needed)
# df = df.drop(columns=['Time', 'Count'])

# # Encode the Zone column to numerical values
# label_encoder = LabelEncoder()
# df['Zone'] = label_encoder.fit_transform(df['Zone'])

# # Split data into training and testing sets
# train, test = train_test_split(df, test_size=0.2, random_state=42)

# # Separate features and target variable from training dataset
# X_train = train.drop(columns='Zone')
# y_train = train['Zone']

# # Separate features and target variable from testing dataset
# X_test = test.drop(columns='Zone')
# y_test = test['Zone']

# # Create the model and fit it to the training data
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# Drop the original Time column and Count column (assuming it's not needed)
df = df.drop(columns=['Time', 'Count'])

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

# Create the SVM model and fit it to the training data
model = SVC(random_state=42, probability=True)  # Set probability=True for soft voting
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
