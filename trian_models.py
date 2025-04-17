import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the dataset
df = pd.read_csv('vellore.csv')

# Rename columns to match our model expectations
df = df.rename(columns={
    'temp_avg': 'tempC',
    'wind_speed': 'windspeedKmph',
    'time': 'date_time'
})

# Convert 'date_time' to datetime object
df['date_time'] = pd.to_datetime(df['date_time'])

# Extract time-based features
df['dayofyear'] = df['date_time'].dt.dayofyear
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday

# Add additional weather features
features = ['dayofyear', 'month', 'weekday', 'temp_min', 'temp_max', 'pressure', 'precipitation']
X = df[features]

# Optional: drop rows with missing values
df.dropna(subset=['tempC', 'windspeedKmph'] + features, inplace=True)

# 1. Temperature prediction
y_temp = df['tempC']
temp_model = LinearRegression()
temp_model.fit(X, y_temp)
print("Feature names:", temp_model.feature_names_in_)
joblib.dump(temp_model, 'temp_model.pkl')

# 2. Wind Speed prediction
y_wind = df['windspeedKmph']
wind_model = LinearRegression()
wind_model.fit(X, y_wind)
joblib.dump(wind_model, 'wind_model.pkl')

# 3. Climate prediction based on temperature and precipitation
def label_climate(row):
    if row['precipitation'] > 0:
        return 'Rainy'
    elif row['tempC'] > 30:
        return 'Hot'
    elif row['tempC'] < 20:
        return 'Cool'
    else:
        return 'Moderate'

df['climate'] = df.apply(label_climate, axis=1)
le = LabelEncoder()
df['climate_encoded'] = le.fit_transform(df['climate'])

y_climate = df['climate_encoded']
climate_model = DecisionTreeClassifier()
climate_model.fit(X, y_climate)

joblib.dump(climate_model, 'climate_model.pkl')
joblib.dump(le, 'climate_encoder.pkl')

print("All models trained and saved successfully!")
