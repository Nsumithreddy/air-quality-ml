🌫️ Air Quality ML
Air Quality ML is a machine learning-based project designed to predict air pollution levels, specifically the Air Quality Index (AQI), using environmental factors such as temperature, humidity, and particulate matter levels. The project includes training models, evaluating their performance, and providing a user-friendly interface through a Streamlit web application.

📖 Project Description
This project uses various regression algorithms to estimate air quality from environmental and pollutant data. The main goal is to build a reliable predictor that can estimate AQI in real-time. It demonstrates how machine learning can be applied to environmental monitoring and public health awareness.

The project includes:
Data collection and preprocessing
Model training and evaluation using scikit-learn and XGBoost
Visualization of AQI trends
A simple web interface using Streamlit to input values and display predictions
Real-time weather data fetched using OpenWeatherMap API

🎯 Objectives
Build multiple regression models to predict AQI values.
Evaluate models using common metrics (MAE, RMSE, R²).
Provide a web interface for users to interact with the predictor.
Visualize AQI data and predictions with graphs.
Predict AQI for the next 4 hours using real-time data.

⚙️ Technologies Used
Python 3
Scikit-learn
XGBoost
Streamlit
OpenWeatherMap API
Pandas
Matplotlib / Seaborn / Plotly
HTML/CSS
Jupyter Notebooks

📁 Project Structure
air-quality-ml/
├── streamlit_app.py → Streamlit application file
├── templates/ → HTML templates (optional)
├── static/ → CSS, JS, images (optional)
├── datasets/ → Raw or preprocessed data files
├── notebooks/ → Jupyter notebooks for training and testing models
│ ├── Random Forest, XGBoost and KNN Regressor.ipynb
│ └── Ridge and Lasso.ipynb
├── models/ → Saved trained model files (.pkl)
├── Plot_AQI.py → Script to generate AQI visualizations
├── requirements.txt → Python package dependencies
└── README.md → Project documentation

🧠 Machine Learning Models
Linear Regression
Ridge Regression
Lasso Regression
K-Nearest Neighbors (KNN)
Random Forest Regressor
XGBoost Regressor

📊 Evaluation Metrics
Each model is evaluated using the following metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

Example performance:
| Model            | R² Score | MAE  | RMSE  |
|------------------|----------|------|-------|
| Random Forest    | 0.85     | 10.3 | 14.6  |
| XGBoost          | 0.83     | 11.0 | 15.2  |
| Ridge Regression | 0.71     | 13.5 | 18.1  |

🖥️ How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/Nsumithreddy/air-quality-ml.git
cd air-quality-ml
Install dependencies
pip install -r requirements.txt
Run the Streamlit app

streamlit run streamlit_app.py
Visit the web interface
Open your browser and go to: http://localhost:8501/

📈 Visualizations
AQI vs Time graphs using Plot_AQI.py
Model performance plots in Jupyter notebooks
Live prediction display in the web interface using Streamlit and Plotly

🔮 Future Enhancements
Integration with more real-time AQI APIs (e.g., AQICN)
City-wise predictions using dropdown
Mobile-friendly responsive UI
Cloud deployment (Streamlit Cloud, Render, Railway, Heroku)

📜 License
This project is licensed under the MIT License. Feel free to use and modify it for personal or academic purposes.

🤝 Contact
Created by Nsumith Reddy
📬 GitHub: https://github.com/Nsumithreddy
