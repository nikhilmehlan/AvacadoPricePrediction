# Avocado Price Prediction

## 📌 Project Overview
Avocados have gained popularity due to their nutritional benefits, leading to increased demand and price fluctuations. This project utilizes machine learning models to predict avocado prices based on historical sales data, geographical variations, and seasonal factors. By leveraging predictive analytics, the project provides insights to help optimize distribution strategies and manage supply chain inefficiencies.

## 📊 Dataset
The dataset includes:
- **Date**: Time of the sale.
- **Region**: Geographical area.
- **Type**: Conventional or organic.
- **Total Volume**: Avocados sold.
- **Average Price**: Price per avocado.
- **Other Sales Metrics**: Total bags, small/large/extra-large bags.

## 🔥 Machine Learning Models Used
1. **XGBoost (Best Model - 89.7% Accuracy)**
   - Gradient boosting method known for efficiency and predictive power.
   - Tuned hyperparameters: learning rate, max depth, estimators.
2. **Random Forest (87% Accuracy)**
   - Ensemble of decision trees reducing overfitting.
3. **Support Vector Machines (82.4% Accuracy)**
   - Utilizes SVR with kernel functions to capture complex relationships.
4. **Decision Tree (81.19% Accuracy)**
   - Simple, interpretable model with non-linear decision boundaries.
5. **k-Nearest Neighbors (78.2% Accuracy)**
   - Instance-based learning using neighbors’ outputs for regression.
6. **Logistic Regression (67.5% Accuracy)**
   - Adapted for regression but limited in handling continuous variables.
7. **Multiple Linear Regression (64.7% Accuracy)**
   - Baseline model assuming linear relationships.

## 🏗 Model Development Workflow
1. **Data Preprocessing**
   - Handled missing values via imputation.
   - Encoded categorical features using one-hot encoding.
   - Normalized numerical features for consistency.
2. **Exploratory Data Analysis (EDA)**
   - Analyzed correlation between features.
   - Visualized trends using time-series plots.
3. **Feature Engineering**
   - Created new features (rolling averages, seasonal trends).
4. **Train-Test Split**
   - 80:20 ratio for training and evaluation.
5. **Model Training & Evaluation**
   - Hyperparameter tuning using GridSearchCV.
   - Performance metrics: Accuracy, MAE, MSE, R² Score.

## 🛠 Tech Stack
- **Programming Language**: Python 3.8+
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Model Deployment: `joblib`, `pickle`
- **Development Tools**: Jupyter Notebook, PyCharm, VS Code
- **Version Control**: Git, GitHub

## 💡 Key Insights
- **XGBoost was the most accurate model (89.7%)**, effectively handling non-linearity.
- **Feature importance analysis** showed that region, type, and seasonality significantly impacted avocado prices.
- **Data-driven decision-making** can optimize production and supply chain management in the avocado industry.

## 📸 Visualizations & Results
- Average avocado prices across cities.
- Total avocado volume by year and type.
- Predicted vs. actual price comparison for each model.

## 🚀 Future Enhancements
- Incorporate external factors (weather conditions, economic indicators).
- Deploy the model via a web application (e.g., Flask, Streamlit).
- Utilize deep learning models for improved accuracy.

## 🔗 Repository
[GitHub Repository](https://github.com/utkarsh369gupta/avacadopriceprediction.git)

## 👨‍💻 Author
**Nikhil Mehlan** | B. Tech – Information Technology | VIT | 22BIT0244

## 📢 Acknowledgments
Guided by **Balakrushna Tripathy**, School of Computer Science Engineering and Information Systems, VIT.

---
Feel free to contribute or provide suggestions to improve this project!

