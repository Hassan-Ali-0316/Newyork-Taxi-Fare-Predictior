import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Load data
df = pd.read_csv('C:/Users/ha159/OneDrive/Documents/TaxiFare prediction/taxi_fare/train.csv')
df = df.drop(columns=['surge_applied'])

# Data preprocessing function
def data_preprocessing(df):
    print(df.head())
    print(df.shape)
    print(df.info())
    x = df.isna().sum()  # Checking for missing values
    if x.sum() > 0:
        df = df.dropna()
    
    y = df.duplicated().sum()  # Checking for duplicate values
    if y != 0:
        df.drop_duplicates(inplace=True)
    
    return df

# Data visualization function
def data_visualization(df):
    sns.heatmap(df.corr(), annot=True)
    plt.show()  # Show heatmap

    for col in df.columns:
        if col != 'total_fare':
            sns.scatterplot(x=df['total_fare'], y=df[col])
            plt.show()

# Model training function
def train(df):
    y = df['total_fare']
    X = df.drop('total_fare', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions for evaluation
    predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    print(f"R^2 Score: {r2}")
    
    evaluation(y_test, predictions)  

    return model, scaler

# Evaluation function
def evaluation(y_test, predictions):
    plt.scatter(y_test, predictions, alpha=0.5)  
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Linear Regression - Actual vs. Predicted Charges')
    plt.show()

# Save model and scaler
def save_model(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

# Load model and scaler
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Predict function
def predict(features, model, scaler):
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

# Preprocess data
#df = data_preprocessing(df)

# Train model and get scaler
#model, scaler = train(df)

# Save the model and scaler
#save_model(model, 'model.pkl')
#save_model(scaler, 'scaler.pkl')
