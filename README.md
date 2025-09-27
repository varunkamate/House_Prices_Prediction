# House_Prices_Prediction

App Live Link:==> https://housepricesprediction-czy6ytqbuwhq79rudzpq52.streamlit.app/

Key Components
Load_Train_Data.ipynb: A Jupyter notebook for data loading, exploratory data analysis, and model training. It compares several regression models and selects the best-performing one. The notebook shows that the CatBoostRegressor was chosen as the best model, achieving an accuracy of 99.31%.

data.csv: The dataset used for training the model. It contains features such as date, price, bedrooms, bathrooms, and sqft_living.

house_price_best_model.pkl: The trained machine learning model, which is a CatBoostRegressor. This pickled file is used by the Streamlit application for making predictions.

app.py: The main application file that uses Streamlit to create an interactive web interface. It allows users to input house features and get a price prediction. It also includes an analysis dashboard to visualize the data.

requirements.txt: A list of all the Python libraries required to run the project, including streamlit, pandas, and catboost.
