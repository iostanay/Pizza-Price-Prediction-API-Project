import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample pizza data
np.random.seed(42)
n_samples = 100

# Features: size (inches), number of toppings
sizes = np.random.uniform(8, 20, n_samples)
num_toppings = np.random.randint(0, 6, n_samples)

# Calculate price with some noise
base_price = 8
price_per_inch = 1.5
price_per_topping = 2
noise = np.random.normal(0, 2, n_samples)
prices = base_price + (sizes * price_per_inch) + (num_toppings * price_per_topping) + noise

# Create DataFrame
pizza_data = pd.DataFrame({
    'size': sizes,
    'num_toppings': num_toppings,
    'price': prices
})

# Prepare features and target
X = pizza_data[['size', 'num_toppings']]
y = pizza_data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance and coefficients
print("Model Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print("\nModel Coefficients:")
print(f"Intercept: ${model.intercept_:.2f}")
print(f"Size coefficient: ${model.coef_[0]:.2f} per inch")
print(f"Toppings coefficient: ${model.coef_[1]:.2f} per topping")

# Function to predict price for new pizzas
def predict_pizza_price(size, num_toppings):
    return model.predict([[size, num_toppings]])[0]

# Example predictions
print("\nExample Predictions:")
test_cases = [
    (12, 2),
    (16, 4),
    (10, 1)
]

for size, toppings in test_cases:
    predicted_price = predict_pizza_price(size, toppings)
    print(f"{size}\" pizza with {toppings} toppings: ${predicted_price:.2f}")

# Visualize the relationship between size and price (holding toppings constant)
plt.figure(figsize=(10, 6))
plt.scatter(X_test['size'], y_test, color='green', label='Actual Prices')
plt.scatter(X_test['size'], y_pred, color='red', label='Predicted Prices')
plt.xlabel('Pizza Size (inches)')
plt.ylabel('Price ($)')
plt.title('Pizza Size vs. Price')
plt.legend()
plt.grid(True)
plt.show()
