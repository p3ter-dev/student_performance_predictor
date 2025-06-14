import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

data = {
    'study_hours': np.random.randint(0, 15, 100),
    'attendance': np.random.uniform(50, 100, 100),
    'current_gpa': np.random.uniform(2.0, 4.0, 100)
}
df = pd.DataFrame(data)

df['gpa_raw'] = (
    0.3 * df['study_hours'] +
    0.3 * (df['attendance']) +
    0.4 * (df['current_gpa'] * 25)
)

gpa_scaler = MinMaxScaler(feature_range=(1.0, 4.0))
df['gpa'] = gpa_scaler.fit_transform(df[['gpa_raw']])

X = df[['study_hours', 'attendance', 'current_gpa']]
y = df['gpa']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'app/model/model.pkl')
joblib.dump(scaler, 'app/model/scaler.pkl')
joblib.dump(gpa_scaler, 'app/model/gpa_scaler.pkl')

print("Model and scalers saved to app/model/")
