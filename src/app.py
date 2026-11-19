from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
from pathlib import Path

app = FastAPI(
    title="Telco Churn Prediction & Retention Engine API",
    description="Production API stub for predicting customer churn risk and delivering personalized retention recommendations.",
    version="1.0.0"
)

# Global variable placeholders for loaded artifacts
fe_pipeline = None
model = None

@app.on_event("startup")
def load_artifacts():
    global fe_pipeline, model
    try:
        model_dir = Path(__file__).resolve().parent.parent / 'models'
        fe_pipeline = joblib.load(model_dir / 'fe_pipeline.joblib')
        model = joblib.load(model_dir / 'best_gb.joblib')  # Tuned Gradient Boosting artifact
        print("Artifacts successfully loaded on startup!")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class CustomerInput(BaseModel):
    Gender: str = Field(..., example="Female")
    Senior_Citizen: str = Field(..., alias="Senior Citizen", example="No")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    Tenure_Months: int = Field(..., alias="Tenure Months", example=1)
    Phone_Service: str = Field(..., alias="Phone Service", example="Yes")
    Multiple_Lines: str = Field(..., alias="Multiple Lines", example="No")
    Internet_Service: str = Field(..., alias="Internet Service", example="Fiber Optic")
    Online_Security: str = Field(..., alias="Online Security", example="No")
    Online_Backup: str = Field(..., alias="Online Backup", example="No")
    Device_Protection: str = Field(..., alias="Device Protection", example="No")
    Tech_Support: str = Field(..., alias="Tech Support", example="No")
    Streaming_TV: str = Field(..., alias="Streaming TV", example="No")
    Streaming_Movies: str = Field(..., alias="Streaming Movies", example="No")
    Contract: str = Field(..., example="Month-to-month")
    Paperless_Billing: str = Field(..., alias="Paperless Billing", example="Yes")
    Payment_Method: str = Field(..., alias="Payment Method", example="Electronic check")
    Monthly_Charges: float = Field(..., alias="Monthly Charges", example=70.05)
    Total_Charges: float = Field(..., alias="Total Charges", example=70.05)

    class Config:
        populate_by_name = True

def get_recommendation_logic(contract, internet_service, tech_support, tenure_months):
    if contract == 'Month-to-month':
        return "Offer long-term contract discount (e.g., 1-Year or 2-Year plan with 15% off)"
    elif internet_service != 'No' and tech_support == 'No':
        return "Proactive Tech Support setup outreach & free 3-month trial of Premium Support"
    elif tenure_months < 12:
        return "First-year loyalty appreciation incentive & check-in call"
    else:
        return "Personalized account review with tailored bundle discount plan"

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if fe_pipeline is None or model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded properly.")
    
    try:
        # Convert inputs to DataFrame with correct original feature column names
        data_dict = {
            'Gender': [customer.Gender],
            'Senior Citizen': [customer.Senior_Citizen],
            'Partner': [customer.Partner],
            'Dependents': [customer.Dependents],
            'Tenure Months': [customer.Tenure_Months],
            'Phone Service': [customer.Phone_Service],
            'Multiple Lines': [customer.Multiple_Lines],
            'Internet Service': [customer.Internet_Service],
            'Online Security': [customer.Online_Security],
            'Online Backup': [customer.Online_Backup],
            'Device Protection': [customer.Device_Protection],
            'Tech Support': [customer.Tech_Support],
            'Streaming TV': [customer.Streaming_TV],
            'Streaming Movies': [customer.Streaming_Movies],
            'Contract': [customer.Contract],
            'Paperless Billing': [customer.Paperless_Billing],
            'Payment Method': [customer.Payment_Method],
            'Monthly Charges': [customer.Monthly_Charges],
            'Total Charges': [customer.Total_Charges]
        }
        df_input = pd.DataFrame(data_dict)
        
        # Preprocess input using loaded pipeline
        df_proc = fe_pipeline.transform(df_input)
        
        # Predict probability
        proba = float(model.predict_proba(df_proc)[0, 1])
        
        # Segment risk
        if proba < 0.3:
            risk_segment = "Low Risk"
        elif proba < 0.7:
            risk_segment = "Medium Risk"
        else:
            risk_segment = "High Risk"
            
        # Get recommendation
        rec = get_recommendation_logic(
            customer.Contract,
            customer.Internet_Service,
            customer.Tech_Support,
            customer.Tenure_Months
        )
        
        return {
            "churn_probability": proba,
            "risk_segment": risk_segment,
            "personalized_recommendation": rec
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the Telco Churn Prediction API! Please visit /docs for API schema definitions."}
