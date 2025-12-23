import os
import uvicorn
import logging
from fastapi import FastAPI, Request
import pandas as pd
from neuralforecast import NeuralForecast
# Explicitly import NHITS to ensure it's available for unpickling
from neuralforecast.models import NHITS
from neuralforecast.core import MODEL_FILENAME_DICT

# Patch MODEL_FILENAME_DICT to handle uppercase 'NHITS'
# The saved model seems to reference 'NHITS' but the registry has 'nhits'
if 'NHITS' not in MODEL_FILENAME_DICT:
    MODEL_FILENAME_DICT['NHITS'] = NHITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    # Vertex AI sets AIP_STORAGE_URI to the path where model artifacts are downloaded
    base_path = os.environ.get("AIP_STORAGE_URI", "nhits_model")
    
    logger.info(f"Base model path from env: {base_path}")
    
    # 1. Use the base path directly
    # We rely on NeuralForecast to handle the GCS path or local path
    actual_model_path = base_path
    
    # Debug logging for local paths only (GCS listing can be flaky with permissions)
    if not base_path.startswith("gs://") and os.path.exists(base_path):
        logger.info(f"Listing contents of {base_path}:")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                full_path = os.path.join(root, file)
                logger.info(f"Found file: {full_path}")
    elif not base_path.startswith("gs://"):
        logger.warning(f"Local base path {base_path} does not exist!")

    # 2. Load the model
    logger.info(f"Attempting to load model from: {actual_model_path}")
    try:
        # NeuralForecast.load expects the directory containing the saved model
        model = NeuralForecast.load(path=actual_model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model from {actual_model_path}: {e}")
        # Crash the app if model fails to load so Vertex AI knows deployment failed
        raise e

@app.get("/health")
def health_check():
    # Simple health check for Vertex AI
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    global model
    if not model:
        logger.error("Predict called but model is not loaded.")
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
        
        # Handle Future Exogenous Variables
        # We assume the client sends History + Future rows.
        # We split the dataframe based on the model's horizon.
        
        # Get the first model (we assume only one model is loaded for serving)
        inner_model = model.models[0]
        horizon = inner_model.h
        
        # Check if model uses future exogenous variables
        uses_future_exog = hasattr(inner_model, 'futr_exog_list') and inner_model.futr_exog_list and len(inner_model.futr_exog_list) > 0
        
        if uses_future_exog:
            # Split into history and future
            # The last 'horizon' rows are treated as future
            if len(df) <= horizon:
                 return {"error": f"Input length ({len(df)}) must be greater than horizon ({horizon}) when using future exogenous variables."}
            
            hist_df = df.iloc[:-horizon].reset_index(drop=True)
            futr_df = df.tail(horizon).reset_index(drop=True)
            
            logger.info(f"Predicting with Future Exog. History: {len(hist_df)}, Future: {len(futr_df)}")
            forecast = model.predict(df=hist_df, futr_df=futr_df)
        else:
            # Standard prediction
            forecast = model.predict(df=df)
        
        # Return results
        return {"predictions": forecast.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Vertex AI sets AIP_HTTP_PORT
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
