from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict
import uvicorn

# Global variables for model and data
model = None
X_test = None
y_test = None
pizza_data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load model and data
    global model, X_test, y_test, pizza_data
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    sizes = np.random.uniform(8, 20, n_samples)
    num_toppings = np.random.randint(0, 6, n_samples)
    
    base_price = 8
    price_per_inch = 1.5
    price_per_topping = 2
    noise = np.random.normal(0, 2, n_samples)
    prices = base_price + (sizes * price_per_inch) + (num_toppings * price_per_topping) + noise
    
    pizza_data = pd.DataFrame({
        'size': sizes,
        'num_toppings': num_toppings,
        'price': prices
    })
    
    # Split and train model
    X = pizza_data[['size', 'num_toppings']]
    y = pizza_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    yield  # Run the FastAPI application
    
    # Cleanup (if needed)
    model = None
    X_test = None
    y_test = None
    pizza_data = None

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Pizza Price Prediction API",
    description="Predict pizza prices using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Data Models
class PizzaInput(BaseModel):
    size: float = Field(..., gt=0, le=30, description="Pizza size in inches")
    num_toppings: int = Field(..., ge=0, le=10, description="Number of toppings")

class PizzaPrediction(BaseModel):
    size: float
    num_toppings: int
    predicted_price: float
    confidence_score: float

class ModelMetrics(BaseModel):
    r2_score: float
    rmse: float
    coefficient_size: float
    coefficient_toppings: float
    intercept: float

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Pizza Price Prediction API",
        "endpoints": {
            "/predict": "Predict price for a single pizza",
            "/batch-predict": "Predict prices for multiple pizzas",
            "/model-metrics": "Get model performance metrics",
            "/visualization": "Get price visualization plot",
            "/data-summary": "Get summary statistics of training data"
        }
    }

@app.post("/predict", response_model=PizzaPrediction)
async def predict_price(pizza: PizzaInput):
    """Predict price for a single pizza"""
    try:
        prediction = model.predict([[pizza.size, pizza.num_toppings]])[0]
        confidence = calculate_confidence_score(pizza)
        
        return PizzaPrediction(
            size=pizza.size,
            num_toppings=pizza.num_toppings,
            predicted_price=round(float(prediction), 2),
            confidence_score=round(float(confidence), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(pizzas: List[PizzaInput]):
    """Predict prices for multiple pizzas"""
    predictions = []
    for pizza in pizzas:
        prediction = model.predict([[pizza.size, pizza.num_toppings]])[0]
        confidence = calculate_confidence_score(pizza)
        predictions.append({
            "size": pizza.size,
            "num_toppings": pizza.num_toppings,
            "predicted_price": round(float(prediction), 2),
            "confidence_score": round(float(confidence), 2)
        })
    return {"predictions": predictions}

@app.get("/model-metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics"""
    y_pred = model.predict(X_test)
    return ModelMetrics(
        r2_score=round(r2_score(y_test, y_pred), 3),
        rmse=round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        coefficient_size=round(float(model.coef_[0]), 3),
        coefficient_toppings=round(float(model.coef_[1]), 3),
        intercept=round(float(model.intercept_), 3)
    )

@app.get("/visualization")
async def get_visualization():
    """Generate and return a visualization plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(pizza_data['size'], pizza_data['price'], c=pizza_data['num_toppings'], 
                cmap='viridis', alpha=0.6)
    plt.colorbar(label='Number of Toppings')
    plt.xlabel('Pizza Size (inches)')
    plt.ylabel('Price ($)')
    plt.title('Pizza Price vs. Size and Toppings')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/data-summary")
async def get_data_summary():
    """Get summary statistics of the training data"""
    return {
        "size_stats": pizza_data['size'].describe().to_dict(),
        "toppings_stats": pizza_data['num_toppings'].describe().to_dict(),
        "price_stats": pizza_data['price'].describe().to_dict(),
        "correlations": pizza_data.corr()['price'].to_dict()
    }

def calculate_confidence_score(pizza: PizzaInput) -> float:
    """Calculate a confidence score for the prediction"""
    size_mean = pizza_data['size'].mean()
    size_std = pizza_data['size'].std()
    toppings_mean = pizza_data['num_toppings'].mean()
    toppings_std = pizza_data['num_toppings'].std()
    
    size_z_score = abs((pizza.size - size_mean) / size_std)
    toppings_z_score = abs((pizza.num_toppings - toppings_mean) / toppings_std)
    
    confidence = 1 - min(1, (size_z_score + toppings_z_score) / 4)
    return confidence

# Save this file as 'pizza_api.py' and run with:
if __name__ == "__main__":
    uvicorn.run("pizza-price-api:app", host="0.0.0.0", port=8000, reload=True)
