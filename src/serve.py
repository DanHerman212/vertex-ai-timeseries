import os
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
from neuralforecast import NeuralForecast

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    # Vertex AI sets AIP_STORAGE_URI to the path where model artifacts are downloaded
    model_path = os.environ.get("AIP_STORAGE_URI", "nhits_model")
    
    print(f"Loading model from {model_path}...")
    try:
        # NeuralForecast.load expects the directory containing the saved model
        model = NeuralForecast.load(path=model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In production, we might want to crash if model fails to load
        # raise e

@app.get("/health")
def health_check():
    # Simple health check for Vertex AI
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    global model
    if not model:
        return {"error": "Model not loaded"}
    
    try:
        body = await request.json()
        
        # Vertex AI sends data in {"instances": [...]} format
        instances = body.get("instances")
        if not instances:
            # Fallback if raw list is sent
            instances = body
            
        if not isinstance(instances, list):
             return {"error": "Input must be a list of records or {'instances': [...]}"}

        # Convert to DataFrame
        df = pd.DataFrame(instances)
        
        # Ensure 'ds' is datetime
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            
        # NeuralForecast predict requires a dataframe with history.
        # It will predict 'h' steps into the future for each unique_id found in df.
        forecast = model.predict(df=df)
        
        # Return results
        return {"predictions": forecast.to_dict(orient="records")}
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Vertex AI sets AIP_HTTP_PORT
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
