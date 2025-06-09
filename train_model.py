import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

data = {
    'study_hours': np.random.randint(0, 15, 100),
    'attendance_rate': np.random.uniform(50, 100, 100),
    'previous_scores_average': np.random.uniform(60, 100, 100),
}
df = pd.DataFrame(data)
df['gpa'] = (
    0.3 * df['study_hours'] +
    0.4 * df['attendance_rate'] +
    0.3 * df['previous_scores_average']
) / 100 * 4

X = df[['study_hours', 'attendance_rate', 'previous_scores_average']]
y = df['gpa']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'app/model/model.pkl')
joblib.dump(scaler, 'app/model/scaler.pkl')

print("Model and scaler saved to app/model/")
