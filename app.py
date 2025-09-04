from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mlflow.pyfunc
import pandas as pd
import uvicorn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model from MLflow registry
model_name = "RF_Final"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

# FastAPI app
app = FastAPI()

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Store: int = Form(...),
    Dept: int = Form(...),
    IsHoliday: int = Form(...),
    MarkDown1: float = Form(...),
    MarkDown2: float = Form(...),
    MarkDown3: float = Form(...),
    MarkDown4: float = Form(...),
    MarkDown5: float = Form(...),
    Type: str = Form(...),
    Size: int = Form(...),
    Week: int = Form(...),
    Month: int = Form(...)
):
    # Prepare input data
    input_data = pd.DataFrame([{
        "Store": Store,
        "Dept": Dept,
        "IsHoliday": IsHoliday,
        "MarkDown1": MarkDown1,
        "MarkDown2": MarkDown2,
        "MarkDown3": MarkDown3,
        "MarkDown4": MarkDown4,
        "MarkDown5": MarkDown5,
        "Type": Type,
        "Size": Size,
        "Week": Week,
        "Month": Month
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Re-render index.html with prediction result
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": round(float(prediction), 2)}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
