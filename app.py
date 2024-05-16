import gradio as gr
import pandas as pd
from sklearn.externals import joblib


model_rfr = joblib.load('model_RFR.pkl')
model_lr = joblib.load('model_LR.pkl')

# Load DataFrame
df = pd.read_csv("Current_Pro_meta.csv")  

def predict(hero_names):
    # Filter heroes
    filtered_heroes = df[df['Name'].isin(hero_names)]
    new_heroes = filtered_heroes[['Primary Attribute', 'Roles', 'Total Pro wins', 'Times Picked', 'Times Banned', 'Win Rate', 'Niche Hero?']]
    new_heroes = new_heroes.dropna()

    # Predict using Linear Regression model
    predictions_LR = model_lr.predict(new_heroes)
    average_win_rate_LR = sum(predictions_LR) / len(predictions_LR)

    # Predict using Random Forest Regressor model
    predictions_RFR = model_rfr.predict(new_heroes)
    average_win_rate_RFR = sum(predictions_RFR) / len(predictions_RFR)

    return {"Linear Regression": average_win_rate_LR, "Random Forest Regressor": average_win_rate_RFR}


hero_names_input = gr.inputs.CheckboxGroup(hero_names, label="Select Hero Names")

interface = gr.Interface(
    fn=predict,
    inputs=hero_names_input,
    outputs=["number", "number"],
    interpretation="default",
    title="Hero Win Rate Prediction",
    description="Predict win rate using Linear Regression and Random Forest Regressor models."
)

interface.launch()
