from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open('dictionary_of_columns_with_categorical_to_index.pkl', 'rb') as file:
    dictionary_of_columns_with_categorical_to_index = pickle.load(file)

with open('xgboost-SMOTE.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('dt-SMOTE.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

def preprocess_data(data):
    df_new = pd.DataFrame([data])
    df_new.replace(dictionary_of_columns_with_categorical_to_index, inplace=True)
    
    if 'gender' in df_new.columns:
        df_new['gender_F'] = (df_new['gender'] == 'Female').astype(int)
        df_new['gender_M'] = (df_new['gender'] == 'Male').astype(int)
        df_new.drop('gender', axis=1, inplace=True)
    
    input_dict = df_new.to_dict(orient='records')[0]
    for key, value in input_dict.items():
        if isinstance(value, pd.Series):
            input_dict[key] = value.iloc[0]
    return pd.DataFrame([input_dict])

def get_prediction(data):
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data.columns)

    dt_prediction = decision_tree_model.predict(preprocessed_data)
    dt_probability = decision_tree_model.predict_proba(preprocessed_data)

    xgb_prediction = xgb_model.predict(preprocessed_data)
    xgb_probability = xgb_model.predict_proba(preprocessed_data)
    
    #X = preprocessed_data.values
    
    #knn_prediction = knn_model.predict(preprocessed_data)
    #knn_probability = knn_model.predict_proba(preprocessed_data)

    #rf_prediction = random_forest_model.predict(preprocessed_data)
    #rf_probability = random_forest_model.predict_proba(preprocessed_data)
    
    probabilities = {
        'XGBoost': float(xgb_probability[0][1]),
        'Decision Tree': float(dt_probability[0][1]),
        #'Random Forest': float(rf_probability[0][1]),
        #'K-nearest Neighbors': float(knn_probability[0][1])
    }

    return probabilities

@app.post("/predict")
async def predict(data: dict):
    probabilities = get_prediction(data)
    return {"probabilities": probabilities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
