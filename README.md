# Pizza Price Prediction API 🍕

A machine learning-powered FastAPI application that predicts pizza prices based on size and number of toppings. The API uses Linear Regression to provide accurate price predictions and includes features like batch prediction, model metrics, and data visualization.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)

## Features
- 🔮 Real-time pizza price predictions
- 📊 Interactive data visualizations
- 📈 Model performance metrics
- 🚀 Batch prediction capabilities
- 📝 Automatic API documentation
- 🎯 Confidence scoring system
- 📊 Summary statistics and data analysis

## Technology Stack
- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy
- matplotlib
- uvicorn
- pydantic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pizza-price-prediction.git
cd pizza-price-prediction
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn pizza-price-api:app --reload
```

2. Open API documentation:
- Navigate to `http://localhost:8000/docs` in your browser
- Interactive API documentation will be available

3. Make predictions:
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"size": 12, "num_toppings": 2}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch-predict",
    json={
        "pizzas": [
            {"size": 12, "num_toppings": 2},
            {"size": 16, "num_toppings": 4}
        ]
    }
)
print(response.json())
```

## API Endpoints

### GET /
- Welcome message and endpoint information
- No parameters required

### POST /predict
- Predicts price for a single pizza
- Parameters:
  - size (float): Pizza size in inches
  - num_toppings (int): Number of toppings

### POST /batch-predict
- Predicts prices for multiple pizzas
- Parameters:
  - List of pizzas with size and num_toppings

### GET /model-metrics
- Returns model performance metrics
- Includes R² score, RMSE, and coefficients

### GET /visualization
- Returns a visualization of price predictions
- Format: PNG image

### GET /data-summary
- Returns summary statistics of training data
- Includes size, toppings, and price statistics

## Model Information

The price prediction model uses Linear Regression with the following features:
- Pizza size (inches)
- Number of toppings

Model performance metrics:
- R² Score: Typically around 0.85-0.90
- RMSE: Approximately $2-3

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Requirements.txt
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
pydantic==2.5.2
```

## Author
[Tanay Bhattacharjee]
- GitHub: [@iostanay]
- LinkedIn: [https://www.linkedin.com/in/tanay-bhattacharjee/]

## Acknowledgments
- FastAPI documentation
- scikit-learn documentation
- Python community
