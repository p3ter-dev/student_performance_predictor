import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path="data/raw/student-mat.csv"):
    df = pd.read_csv(path)

    df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    # Drop original grade columns (optional for classification)
    df = df.drop(['G1', 'G2', 'G3'], axis=1)

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    numerical_cols = df.select_dtypes(include='number').drop(columns=['pass_fail']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split into features and target
    X = df.drop('pass_fail', axis=1)
    y = df['pass_fail']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
