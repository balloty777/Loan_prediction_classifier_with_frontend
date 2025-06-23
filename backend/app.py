from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
from main import MapEncoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class UserInput(BaseModel):
    person_age: Annotated[int, Field(..., gt=0, lt=110)]
    person_gender: Annotated[Literal['male', 'female'], Field(...)]
    person_education: Annotated[Literal['High School', 'Associate', 'Bachelor', 'Master'], Field(...)]
    person_income: Annotated[float, Field(..., gt=0)]
    person_emp_exp: Annotated[int, Field(..., ge=0)]
    person_home_ownership: Annotated[Literal['OWN', 'RENT', 'MORTGAGE', 'OTHER'], Field(...)]
    loan_amnt: Annotated[float, Field(..., gt=0)]
    loan_intent: Annotated[str, Field(...)]
    loan_int_rate: Annotated[float, Field(..., gt=0)]
    cb_person_cred_hist_length: Annotated[float, Field(..., gt=0)]
    credit_score: Annotated[int, Field(..., gt=0)]
    previous_loan_defaults_on_file: Annotated[Literal['YES', 'NO'], Field(...)]

    @computed_field
    @property
    def loan_percent_income(self) -> float:
        return self.loan_amnt / self.person_income

@app.post("/predict")
def predict_loan_approval(user_input: UserInput):
    input_data = user_input.model_dump()
    input_data["loan_percent_income"] = user_input.loan_percent_income
    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    return {"loan_status_prediction": int(prediction)}
