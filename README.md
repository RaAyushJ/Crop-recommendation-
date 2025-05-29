SmartCrop: An Intelligent Crop Recommendation System

SmartCrop is an AI-powered system designed to assist farmers in selecting the most suitable crops based on environmental and soil data. Leveraging machine learning algorithms and explainable AI (XAI) techniques, the system provides transparent and actionable crop recommendations to promote sustainable agriculture.

Table of Contents

- Features

- Dataset

- Technologies Used

- Installation

- Usage

- Modules & Functions

- Explaining the System

- Contributing

- License

- Contact

Features

- Predicts suitable crops based on soil and environmental data

- Utilizes machine learning models: Random Forest, Support Vector Machine, Neural Networks

- Incorporates explainable AI (LIME, SHAP) for transparency

- Interactive visualization of crop nutrient data (N-P-K values)

- Integrates real-time data sources (soil sensors, weather data) (future scope)

- User-friendly interface for farmers with limited technical knowledge

Dataset

- The dataset Crop_recommendation.csv contains:

- Soil nutrients: Nitrogen (N), Phosphorus (P), Potassium (K)

- Environmental factors: temperature, humidity, pH, rainfall

- Target variable: crop labels (e.g., rice, wheat, maize)

- Data is sourced from agricultural departments, research papers, and online resources

Technologies Used

- Python 3.x

- pandas, numpy

- scikit-learn (for machine learning algorithms)

- matplotlib, seaborn, plotly (for visualization)

- shap, lime (for explainable AI)

- Jupyter Notebook (for development & exploration)

Installation

Clone the repository

git clone https://github.com/yourusername/SmartCrop.git
cd SmartCrop

Setup environment

It is recommended to create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Sample requirements.txt content:

pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
shap
lime

Note: For full reproducibility, include the requirements.txt with the exact versions used.

Usage

Data loading and exploration

The code loads the dataset, performs basic data checks, and visualizes environmental and soil data.

import pandas as pd
df = pd.read_csv("Crop_recommendation.csv")
print(df.info())

Model training

Train different machine learning models to predict suitable crops based on the data.

# Example:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

Model explainability

Use SHAP or LIME to interpret model predictions.

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

Visualization

Plot nutrient levels across crops.

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(
x=crop_summary.index,
y=crop_summary['N'],
name='Nitrogen',
marker_color='mediumvioletred'
))
# similar for P and K
fig.show()

Predictions & Recommendations

Use the trained models to predict suitable crops based on user input environmental data.

Explaining the System

The system incorporates XAI techniques to:

- Provide explanations for each crop recommendation

- Increase trust and transparency

- Support farmers in understanding the factors influencing the AI's suggestions

Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

Guidelines:

- Follow PEP8 standards

- Document new features properly

- Include test cases for new functionalities
