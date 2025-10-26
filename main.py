import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn import svm

# Importing the dataset
df = pd.read_excel(r'C:\Users\HusseinEl-Morsy\OneDrive - IBM\Desktop\ML\HousePricePrediction\dataset.xlsx')

df.drop(['Id'], axis=1, inplace=True)
df = df.dropna(subset=df.columns[:-1])

X = df.iloc[:, :-1]
y = df.iloc[:, -1].values

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
y = y.reshape(-1, 1)
imputer.fit(y)
y = imputer.transform(y).ravel()

# Detect categorical & numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Apply OneHotEncoding to categorical & StandardScaler to numerical
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ]
)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Train the Random-Forest Model
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Prediction for the Random-Forest Model
y_pred = reg.predict(X_test)

# Evaluate the Random-Forest model results
r_forest_mae = mean_absolute_percentage_error(y_test, y_pred)
print(f"Random-Forest Mean Absolute Error: {r_forest_mae:.2f}")

# Train the SVR Model
svr = svm.SVR()
svr.fit(X_train, y_train)

# Prediction for the Random-Forest Model
y_pred = svr.predict(X_test)

# Evaluate the SVR model results
svr_mae = mean_absolute_percentage_error(y_test, y_pred)
print(f"SVR Model Mean Absolute Error: {svr_mae:.2f}")

