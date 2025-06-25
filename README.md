🌫️ Air Quality ML
Air Quality ML is a machine learning-based project designed to predict air pollution levels, specifically the Air Quality Index (AQI), using environmental factors such as temperature, humidity, and particulate matter levels. The project includes training models, evaluating their performance, and providing a user-friendly interface through a Streamlit web application.

📖 Project Description
This project uses various regression algorithms to estimate air quality from environmental and pollutant data. The main goal is to build a reliable predictor that can estimate AQI in real-time. It demonstrates how machine learning can be applied to environmental monitoring and public health awareness.

The project includes:
- Data collection and preprocessing
- Model training and evaluation using scikit-learn and XGBoost
- Visualization of AQI trends
- A web interface using Streamlit for live prediction
- Real-time weather data fetched from OpenWeatherMap API

🎯 Objectives
- Build multiple regression models to predict AQI values
- Evaluate models using MAE, RMSE, and R²
- Provide a user interface for real-time AQI prediction
- Visualize AQI data and model performance

⚙️ Technologies Used
- Python 3
- Scikit-learn
- XGBoost
- Streamlit
- OpenWeatherMap API
- Pandas
- Matplotlib, Seaborn, Plotly
- Jupyter Notebooks

📁 Project Structure

air-quality-ml/
├── streamlit\_app.py
├── templates/
├── static/
├── datasets/
├── notebooks/
│   ├── XGBoost
├── models/
├── Plot\_AQI.py
├── requirements.txt
└── README.md

🧠 Machine Learning Models
- XGBoost Regressor

📊 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Example Performance:

| Model             | R² Score | MAE  | RMSE |
|------------------|----------|------|------|
| Random Forest     | 0.85     | 10.3 | 14.6 |
| XGBoost           | 0.83     | 11.0 | 15.2 |
| Ridge Regression  | 0.71     | 13.5 | 18.1 |

🖥️ How to Run

git clone [https://github.com/Nsumithreddy/air-quality-ml.git](https://github.com/Nsumithreddy/air-quality-ml.git)
cd air-quality-ml
pip install -r requirements.txt
streamlit run streamlit\_app.py

Then open your browser at: `http://localhost:8501/`

📈 Visualizations
- AQI trends using Plot_AQI.py
- Model evaluation in Jupyter notebooks
- Interactive real-time prediction with Plotly in Streamlit

🔮 Future Enhancements
- Support for more cities and regions
- Improved UI/UX
- Cloud deployment (Streamlit Cloud, Render, Heroku)
- Integration with additional AQI APIs (e.g., AQICN)

📜 License
This project is licensed under the MIT License.

🤝Contact
Created by Nsumith Reddy  
GitHub: [https://github.com/Nsumithreddy](https://github.com/Nsumithreddy)

📦Languages
- Python: 100%

