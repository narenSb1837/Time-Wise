import io
from io import StringIO
import pandas as pd
import tsfresh as ts
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import plotly

from tensorflow.keras.models import model_from_json

from models import *


templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

def features_extraction(df):
    df['id'] = 1
    df = df.iloc[: , 1:]
    df_rolled = ts.utilities.dataframe_functions.roll_time_series(df,column_id= 'id',
                                                                column_sort='point_timestamp',
                                                                  min_timeshift=24,
                                                                  max_timeshift=24)
    df_features = ts.extract_features(df_rolled, column_id='id', column_sort='point_timestamp', 
                                       default_fc_parameters=ts.feature_extraction.MinimalFCParameters())
    return df_features

def classifier(df_features):
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model1.h5")
    print("Loaded model from disk")
    p = loaded_model.predict(df_features)
    p = p.argmax(axis=1)
    counts = np.bincount(p)
    selected_mod = np.argmax(counts)
    class_dict = {0:'ARIMA',1:'XgBoost',2:'ETS Model',3:'Neural Prophet'}
    print("Classified Model is ",class_dict[selected_mod])
    return selected_mod


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/uploadFile")
async def handle_form(request: Request,input_file: UploadFile = File(...)):
    contents = await input_file.read()
    csv_string = contents.decode('utf-8')  
    df = pd.read_csv(StringIO(csv_string))

    df_features = features_extraction(df)
    selected_mod = classifier(df_features)

    train_size = int(0.8 * len(df))
    # Split the data into training and testing sets
    train_data = df[:train_size][['point_timestamp','point_value']]
    test_data = df[train_size:][['point_timestamp','point_value']]

    if selected_mod == 0:
        graphJSON, stats, summary = arima(train_data,test_data)
    elif selected_mod == 1:
        graphJSON, stats, summary = Xgb(df)
    elif selected_mod == 2:
        graphJSON, stats, summary = ets(train_data,test_data)
    elif selected_mod == 3:
        graphJSON, stats, summary = neural_prophet(train_data,test_data)

    fig = px.line(df,x = 'point_timestamp', y = 'point_value' )
    figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return templates.TemplateResponse("getDateInput.html", {"request": request, "figJSON" : figJSON, "graphJSON" : graphJSON, "ModelName" : stats[0], "MAPE": stats[1]})



