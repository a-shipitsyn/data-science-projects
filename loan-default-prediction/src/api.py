"""REST API: default probability and approve/decline decision for a loan application.

Run from src/:  uvicorn api:app      (model path from the MODEL_PATH variable)
"""
import os
import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import schema

MODEL_PATH = os.environ.get('MODEL_PATH', '../models/model.pkl')
state: dict = {}

EXAMPLE = {
    'issue_date': '2025-06-12', 'amount': 120, 'card_exp_month': 9, 'card_exp_year': 2027,
    'existing_debt': 40, 'active_loans': 1, 'history_days': 410,
    'new_credit_7d': 0, 'new_credit_14d': 0, 'paid_on_time_3m': 3, 'paid_on_time_6m': 5,
    'failed_payments_3m': 0, 'failed_payments_6m': 1, 'failed_payments_1y': 1,
    'repaid_14d': 60, 'repaid_1m': 95, 'repaid_3m': 210, 'repaid_6m': 380, 'repaid_1y': 650,
    'merchant_group': 'fashion', 'merchant_category': 'shoes',
}


def load_model(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state.update(load_model(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title='Loan default prediction', version='1.0', lifespan=lifespan)


class LoanApplication(BaseModel):
    model_config = {'json_schema_extra': {'examples': [EXAMPLE]}}

    issue_date: date
    amount: float = Field(gt=0)
    card_exp_month: int | None = Field(None, ge=1, le=12)
    card_exp_year: int | None = Field(None, ge=2000, le=2100)
    existing_debt: float | None = None       # negative: credit balance
    active_loans: int = Field(ge=0)
    history_days: int = Field(ge=0)
    new_credit_7d: float = Field(ge=0)
    new_credit_14d: float = Field(ge=0)
    paid_on_time_3m: int = Field(ge=0)
    paid_on_time_6m: int = Field(ge=0)
    failed_payments_3m: int = Field(ge=0)
    failed_payments_6m: int = Field(ge=0)
    failed_payments_1y: int = Field(ge=0)
    repaid_14d: float = Field(ge=0)
    repaid_1m: float = Field(ge=0)
    repaid_3m: float = Field(ge=0)
    repaid_6m: float = Field(ge=0)
    repaid_1y: float = Field(ge=0)
    merchant_group: str
    merchant_category: str


class Prediction(BaseModel):
    default_probability: float
    decision: str
    threshold: float
    model: str


@app.get('/')
def health() -> dict:
    return {'status': 'ok', 'model_loaded': 'model' in state}


@app.get('/model')
def model_info() -> dict:
    return {k: v for k, v in state['meta'].items() if k != 'profile'}


@app.post('/predict', response_model=Prediction)
def predict(application: LoanApplication,
            max_pd: float | None = Query(
                None, gt=0, lt=1,
                description='decline at or above this default probability; '
                            'default: the threshold chosen in training')) -> Prediction:
    df = pd.DataFrame([application.model_dump()])
    df[schema.DATE] = df[schema.DATE].astype(str)
    df = df.astype({c: float for c in schema.OPTIONAL})
    try:
        schema.validate(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    p = float(state['model'].predict_proba(df)[0, 1])
    cut = state['model'].threshold_ if max_pd is None else max_pd
    return Prediction(default_probability=round(p, 4),
                      decision='decline' if p >= cut else 'approve',
                      threshold=round(cut, 4), model=state['meta']['model'])
