import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class GeoZonePredictor:
    def _init_(self, clf_estimators):
        self.clf_estimators = clf_estimators
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None

    def preprocess(self, df):
        def process_time(time_str):
            time_parts = time_str.split(' ')[0].split(':')
            hour, minute, second = map(int, time_parts)
            am_pm = 0 if 'AM' in time_str else 1  # 0 for AM, 1 for PM
            return hour, minute, second, am_pm

        df[['Hour', 'Minute', 'Second', 'AM_PM']] = df['Time'].apply(lambda x: pd.Series(process_time(x)))
        df['Hour_sin'] = np.sin(df['Hour'] * (2. * np.pi / 24))
        df['Hour_cos'] = np.cos(df['Hour'] * (2. * np.pi / 24))
        df['Minute_sin'] = np.sin(df['Minute'] * (2. * np.pi / 60))
        df['Minute_cos'] = np.cos(df['Minute'] * (2. * np.pi / 60))
        df = df.drop(columns=['Time', 'Count', 'Hour', 'Minute', 'Second'])
        df['Zone'] = self.label_encoder.fit_transform(df['Zone'])
        return df

    def fit(self, df):
        df = self.preprocess(df)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train.drop(columns='Zone')
        y_train = train['Zone']
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = VotingClassifier(estimators=self.clf_estimators, voting='soft')
        self.model.fit(X_train_scaled, y_train)

    def predict(self, df):
        df = self.preprocess(df)
        X_test = df.drop(columns='Zone')
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

    def evaluate(self, df):
        df = self.preprocess(df)
        X_test = df.drop(columns='Zone')
        y_test = df['Zone']
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)

# Usage:
# Assuming `best_knn` and `best_xgb` are your best models from your code above.
clf_estimators = [('knn', best_knn), ('xgb', best_xgb)]
predictor = GeoZonePredictor(clf_estimators)
df = pd.read_csv('geo_points.csv')
predictor.fit(df)
predictor.evaluate(df)
predictor.save_model('model.pkl')