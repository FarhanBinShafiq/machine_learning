# gradient_descent.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and check the data
try:
    df = pd.read_csv('home data.csv')
    print("First few rows of data:")
    print(df.head())
    print("\nChecking for NaN values:")
    print(df.isna().sum())
except FileNotFoundError:
    print("Error: 'home data.csv' not found. Please ensure the file exists.")
    exit()

# Extract x and y columns
try:
    x = df['x'].values  # Convert to numpy array
    y = df['y'].values  # Convert to numpy array
except KeyError:
    print("Error: 'x' or 'y' column not found in the CSV file.")
    exit()

# Initial parameters
m = 0  # Slope
c = 0  # Intercept
learning_rate = 0.0001  # Reduced learning rate to prevent divergence
n = len(x)
iterations = 1000

# Lists to store history for plotting
m_history = []
c_history = []
cost_history = []

# Gradient Descent with debugging
for i in range(iterations):
    # Predicted values
    y_pre = m * x + c
    
    # Check for NaN in predictions
    if np.any(np.isnan(y_pre)):
        print(f"NaN detected in y_pre at iteration {i}")
        break
    
    # Calculate cost (Mean Squared Error)
    cost = (1/n) * sum((y - y_pre) ** 2)
    cost_history.append(cost)
    
    # Calculate gradients
    der_m = (-2/n) * sum(x * (y - y_pre))  # Partial derivative w.r.t m
    der_c = (-2/n) * sum(y - y_pre)        # Partial derivative w.r.t c
    
    # Check gradients for NaN
    if np.isnan(der_m) or np.isnan(der_c):
        print(f"NaN in gradients at iteration {i}: der_m={der_m}, der_c={der_c}")
        break
    
    # Update parameters
    m = m - learning_rate * der_m
    c = c - learning_rate * der_c
    
    # Store history
    m_history.append(m)
    c_history.append(c)
    
    # Print progress every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: m={m}, c={c}, Cost={cost}")

# Final results
print(f"\nFinal Slope (m): {m}")
print(f"Final Intercept (c): {c}")

# Test prediction
test_x = 50
predicted_y = m * test_x + c
print(f"Prediction for x = {test_x}: {predicted_y}")

# Plotting the results
# 1. Data points and fitted line
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m * x + c, color='red', label=f'Fit: y = {m:.4f}x + {c:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()

# 2. Cost history
plt.subplot(1, 2, 2)
plt.plot(cost_history, color='green')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost Reduction Over Time')

plt.tight_layout()
plt.show()