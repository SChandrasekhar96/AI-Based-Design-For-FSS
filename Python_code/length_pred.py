import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import threading

# Load the dataset (adjust for Excel if needed)
file_path = './Length_variation_Analyze_S21.csv.xls'  # Ensure this path is correct
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Reshape the dataset
reshaped_data = pd.melt(
    data,
    id_vars=['Freq [GHz]'],
    value_vars=[col for col in data.columns if "dB(S(2,1))" in col],
    var_name='Parameter',
    value_name='Value'
)

# Extract length from the parameter names
reshaped_data['Length'] = reshaped_data['Parameter'].apply(lambda x: int(x.split("'")[1].replace('mm', '')))
reshaped_data['Parameter'] = 'S21'

# Pivot to create a DataFrame with columns for Freq, S21, and Length
pivoted_data = reshaped_data.pivot_table(index=['Freq [GHz]', 'Length'], columns='Parameter', values='Value').reset_index()

# Split data into features (X) and target (y)
X = pivoted_data[['Freq [GHz]', 'S21']]
y = pivoted_data['Length']

# Create a pipeline that includes PolynomialFeatures and Ridge regression
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),  # Increase degree for more complexity
    ('ridge', Ridge())
])

# Set up GridSearchCV to tune alpha and degree parameters
param_grid = {
    'poly__degree': [2, 3, 4],  # Try different degrees
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0]  # Try different regularization strengths
}

# Use cross-validation to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Use the best model found
best_model = grid_search.best_estimator_

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model on the training set
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Best Model - MSE: {mse}, R^2: {r2}')

# Function to predict length
def predict_length(frequency, s21_value):
    input_data = pd.DataFrame([[frequency, s21_value]], columns=['Freq [GHz]', 'S21'])
    predicted_length = best_model.predict(input_data)[0]
    return predicted_length

# Function to continuously take user input without closing the plot
def user_input_loop():
    while True:
        try:
            input_freq = float(input("Enter the frequency (GHz) or type 'exit' to quit: "))
            input_s21 = float(input("Enter the S21 value: "))
            predicted_length = predict_length(input_freq, input_s21)
            print(f'Predicted Length: {predicted_length:.2f} mm')
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except:
            print("An error occurred. Exiting...")
            break

        # Check if the user wants to exit
        cont = input("Do you want to predict another length? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting the prediction loop.")
            break

# Generate a grid of frequency and S21 values for the heatmap
freq_values = np.linspace(pivoted_data['Freq [GHz]'].min(), pivoted_data['Freq [GHz]'].max(), 100)
s21_values = np.linspace(pivoted_data['S21'].min(), pivoted_data['S21'].max(), 100)

freq_grid, s21_grid = np.meshgrid(freq_values, s21_values)

# Flatten the grids to pass them into the prediction function
freq_flat = freq_grid.flatten()
s21_flat = s21_grid.flatten()

# Predict lengths for each pair of frequency and S21 in the grid
predicted_length_grid = np.array([predict_length(f, s21) for f, s21 in zip(freq_flat, s21_flat)])
predicted_length_grid = predicted_length_grid.reshape(freq_grid.shape)

# Interpolate actual lengths onto the same grid for comparison
actual_length_grid = griddata(
    (pivoted_data['Freq [GHz]'], pivoted_data['S21']),
    pivoted_data['Length'],
    (freq_grid, s21_grid),
)

# Plot the heatmap for actual lengths
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.contourf(freq_grid, s21_grid, actual_length_grid, levels=100, cmap='viridis')
plt.colorbar(label='Actual Length (mm)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S21 (dB)')
plt.title('Heatmap of Actual Length')

# Plot the heatmap for predicted lengths, limiting to 100mm
plt.subplot(2, 1, 2)
plt.contourf(freq_grid, s21_grid, predicted_length_grid, levels=np.linspace(0, 60, 100), cmap='viridis')
plt.colorbar(label='Predicted Length (mm)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S21 (dB)')
plt.title('Heatmap of Predicted Length')

plt.tight_layout()

# Start the user input loop in a separate thread
input_thread = threading.Thread(target=user_input_loop)
input_thread.start()

# Display the plot (this will not block the thread running the input loop)  
plt.show()

# Ensure the thread continues running until the user input loop is complete
input_thread.join()
