import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Specify the file path
file_path = './dataset.xls'  # Change this to your actual file path

# Load the dataset from an Excel file
df = pd.read_csv('./dataset.xls')# Use pd.read_excel for .xls files

# Display the first few rows of the dataset to understand its structure
print("Dataset preview:")
print(df.head())

# Check if the necessary columns exist
required_columns = ['Freq [GHz]', 'dB(S(2,1)) []']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Features and Target
X = df[['Freq [GHz]']]  # Use the correct column name
y_s21 = df['dB(S(2,1)) []']  # Use the correct column name

# Split data into training and testing sets
X_train, X_test, y_s21_train, y_s21_test = train_test_split(X, y_s21, test_size=0.2, random_state=42)

# Initialize the Ridge Regression model
model_s21 = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization strength

# Train the model
model_s21.fit(X_train, y_s21_train)

# Make predictions
y_s21_pred = model_s21.predict(X_test)

# Evaluate the model
print("\nS21 Prediction")
print("Mean Squared Error:", mean_squared_error(y_s21_test, y_s21_pred))
print("R^2 Score:", r2_score(y_s21_test, y_s21_pred))

# Plot the results
plt.figure(figsize=(6, 6))

# S21 Plot
plt.scatter(X_test, y_s21_test, color='green', label='Actual S21')
plt.plot(X_test, y_s21_pred, color='orange', linewidth=2, label='Predicted S21')
plt.title('Frequency vs S21')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S21')
plt.legend()

plt.tight_layout()  # Adjust layout
plt.show()

# User Input for Prediction
try:
    user_freq = float(input("\nEnter a frequency in GHz to predict S21: "))
    user_freq_array = np.array([[user_freq]])  # Convert to 2D array
    user_s21_pred = model_s21.predict(user_freq_array)
    print(f"\nFor Frequency = {user_freq} GHz:")
    print(f"Predicted S21 = {user_s21_pred[0]}")
except ValueError:
    print("Invalid input. Please enter a numerical value for the frequency.")
