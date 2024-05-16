import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

model_rfr = joblib.load('model_RFR.pkl')
model_lr = joblib.load('model_LR.pkl')
model_lor = joblib.load('model_LoR.pkl')
model_rfc = joblib.load('model_RFC.pkl')

df = pd.read_csv("Current_Pro_meta.csv")

scaler = StandardScaler()

def preprocess_heroes(filtered_heroes):
    filtered_heroes = filtered_heroes[['Primary Attribute', 'Roles', 'Total Pro wins', 'Times Picked', 'Times Banned', 'Win Rate', 'Niche Hero?']]
    
    le = LabelEncoder()
    filtered_heroes.loc[:, 'Primary Attribute'] = le.fit_transform(filtered_heroes['Primary Attribute'])
    filtered_heroes.loc[:, 'Roles'] = le.fit_transform(filtered_heroes['Roles'])

    
    return filtered_heroes

def predict(hero_names):
    filtered_heroes = df[df['Name'].isin(hero_names)]
    
    new_heroes = preprocess_heroes(filtered_heroes)
    
    if new_heroes.empty:
        return 0, 0
    
    try:
        predictions_LR = model_lr.predict(new_heroes)
        average_win_rate_LR = sum(predictions_LR) / len(predictions_LR)
    except Exception as e:
        print("Error in Linear Regression prediction:", e)
        average_win_rate_LR = None
    
    try:
        predictions_RFR = model_rfr.predict(new_heroes)
        average_win_rate_RFR = sum(predictions_RFR) / len(predictions_RFR)
    except Exception as e:
        print("Error in Random Forest Regressor prediction:", e)
        average_win_rate_RFR = None
    
    return average_win_rate_LR, average_win_rate_RFR



def predict_win_rate_LR(file):
    if file is None:
        return "Please upload a CSV file."
    
    df_test_features = pd.read_csv(file.name, index_col='match_id_hash')
    
    X_test = df_test_features.values
    X_test_scaled = scaler.fit_transform(X_test)
    
    y_test_pred = model_lor.predict_proba(X_test_scaled)[:, 1]
    
    df_submission = pd.DataFrame({'match_id_hash': df_test_features.index, 'radiant_win_prob': y_test_pred})
    
    return df_submission

def predict_win_rate_RFC(file):
    if file is None:
        return "Please upload a CSV file."
    
    df_test_features = pd.read_csv(file.name, index_col='match_id_hash')
    
    X_test = df_test_features.values
    
    y_test_pred = model_rfc.predict_proba(X_test)[:, 1]
    
    df_submission = pd.DataFrame({'match_id_hash': df_test_features.index, 'radiant_win_prob': y_test_pred})
    
    return df_submission

hero_names = df['Name'].unique().tolist()

hero_names_input = gr.CheckboxGroup(choices=hero_names, label="Select Hero Names")

file_input_LR = gr.UploadButton(file_types=[".csv"], label="Upload Test Features (CSV)")
file_input_RFC = gr.UploadButton(file_types=[".csv"], label="Upload Test Features (CSV)")


interface = gr.Interface(
    fn=predict,
    inputs=hero_names_input,
    outputs=[gr.Textbox(label="Linear Regression Win Rate"), gr.Textbox(label="Random Forest Regressor Win Rate")],
    title="Hero Win Rate Prediction",
    description="Predict win rate using Linear Regression and Random Forest Regressor models."
)

interface_LR = gr.Interface(
    fn=predict_win_rate_LR,
    inputs=file_input_LR,
    outputs="dataframe",
    title="Predict Win Rate (Linear Regression)",
    description="Upload a CSV file containing test features to predict win rate using Linear Regression model."
)

interface_RFC = gr.Interface(
    fn=predict_win_rate_RFC,
    inputs=file_input_RFC,
    outputs="dataframe",
    title="Predict Win Rate (Random Forest Classifier)",
    description="Upload a CSV file containing test features to predict win rate using Random Forest Classifier model."
)

if __name__ == "__main__":
    gr.TabbedInterface(
        [interface, interface_LR, interface_RFC], ["Hero Win Rate", "Logistic Regression (Match Win Rate)", "Random Forest Classifier (Match Win Rate)"]
    ).launch(debug=True)
