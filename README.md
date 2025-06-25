🌫️ Air Quality ML
Air Quality ML** is a machine learning-based project designed to predict air pollution levels, specifically the Air Quality Index (AQI), using environmental factors such as temperature, humidity, and particulate matter levels. The project includes training models, evaluating their performance, and providing a user-friendly interface through a Flask web application.

📖 Project Description
This project uses various regression algorithms to estimate air quality from environmental and pollutant data. The main goal is to build a reliable predictor that can estimate AQI in real-time. It demonstrates how machine learning can be applied to environmental monitoring and public health awareness.

The project includes:
- Data collection and preprocessing
- Model training and evaluation using scikit-learn and XGBoost
- Visualization of AQI trends
- A simple web interface using Flask to input values and display predictions

🎯 Objectives
- Build multiple regression models to predict AQI values.
- Evaluate models using common metrics (MAE, RMSE, R²).
- Provide a web interface for users to interact with the predictor.
- Visualize AQI data and predictions with graphs.

⚙️ Technologies Used
- Python 3
- Scikit-learn
- XGBoost
- Flask
- Pandas
- Matplotlib / Seaborn
- HTML/CSS (for web UI)
- Jupyter Notebooks

## 📁 Project Structure
air-quality-ml/
│
├── app.py                  → Flask application file
├── templates/              → HTML templates for Flask UI
├── static/                 → CSS, JS, images (optional)
├── datasets/               → Raw or preprocessed data files
├── notebooks/              → Jupyter notebooks for training and testing models
│   ├── Random Forest, XGBoost and KNN Regressor.ipynb
│   └── Ridge and Lasso.ipynb
├── models/                 → Saved trained model files (.pkl)
├── Plot\_AQI.py             → Script to generate AQI visualizations
├── requirements.txt        → Python package dependencies
└── README.md               → Project documentation

🧠 Machine Learning Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors (KNN)
- Random Forest Regressor
- XGBoost Regressor

📊 Evaluation Metrics
Each model is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Example performance:
| Model           | R² Score | MAE   | RMSE  |
|-----------------|----------|-------|--------|
| Random Forest   | 0.85     | 10.3  | 14.6   |
| XGBoost         | 0.83     | 11.0  | 15.2   |
| Ridge Regression| 0.71     | 13.5  | 18.1   |

🖥️ How to Run

1. Clone the repository
   git clone https://github.com/Nsumithreddy/air-quality-ml.git
   >cd air-quality-ml

2. Install dependencies
   >pip install -r requirements.txt

3. Run the Flask app
   > python app.py

4. Visit the web interface
   Open your browser and go to:
   `http://localhost:5000/`

📈 Visualizations
* AQI vs Time graphs using `Plot_AQI.py`
* Model performance plots in Jupyter notebooks
* Live prediction display in the web interface

🔮 Future Enhancements
* Integration with real-time AQI API (e.g., OpenWeather, AQICN)
* City-wise predictions using dropdown
* Mobile-friendly UI
* Cloud deployment (Render, Railway, Heroku)

📜 License
This project is licensed under the **MIT License**. Feel free to use and modify it for personal or academic purposes.

🤝 Contact
Created by **Nsumith Reddy**
📬 GitHub: [https://github.com/Nsumithreddy](https://github.com/Nsumithreddy)

