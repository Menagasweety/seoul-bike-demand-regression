import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Seoul Bike Demand Predictor", page_icon="🚲", layout="wide")
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"processed"/"seoul_bike_clean.csv"
TARGET="rented_bike_count"
FEATURES=["hour","temperature_c","humidity_pct","wind_speed_ms","visibility_10m","dew_point_temperature_c","solar_radiation_mj_m2","rainfall_mm","snowfall_cm","season","holiday","functioning_day","year","month","day_of_week","is_weekend","hour_sin","hour_cos"]
CAT=["season","holiday","functioning_day"]; NUM=[c for c in FEATURES if c not in CAT]

@st.cache_data
def load_data():
    return pd.read_csv(DATA,parse_dates=["date"])

@st.cache_resource
def train_models(df):
    prep=ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
    defs={"Ridge Regression":Ridge(alpha=10.0),"Lasso Regression":Lasso(alpha=0.05,max_iter=20000,random_state=42),"ElasticNet Regression":ElasticNet(alpha=0.05,l1_ratio=0.5,max_iter=20000,random_state=42)}
    split=int(len(df)*0.8); X_train=df.sort_values(["date","hour"]).iloc[:split][FEATURES]; y_train=df.sort_values(["date","hour"]).iloc[:split][TARGET]; X_test=df.sort_values(["date","hour"]).iloc[split:][FEATURES]; y_test=df.sort_values(["date","hour"]).iloc[split:][TARGET]
    metrics=[]; deployed={};
    baseline=np.full(len(y_test),y_train.mean()); metrics.append({"Model":"Mean Baseline","MAE":mean_absolute_error(y_test,baseline),"RMSE":mean_squared_error(y_test,baseline)**0.5,"R²":r2_score(y_test,baseline)})
    for name,est in defs.items():
        pipe=Pipeline([("preprocess",prep),("model",est)]); pipe.fit(df[FEATURES],df[TARGET]); deployed[name]=pipe
        eval_pipe=Pipeline([("preprocess",prep),("model",est.__class__(**est.get_params()))]); eval_pipe.fit(X_train,y_train); p=eval_pipe.predict(X_test)
        metrics.append({"Model":name,"MAE":mean_absolute_error(y_test,p),"RMSE":mean_squared_error(y_test,p)**0.5,"R²":r2_score(y_test,p)})
    return pd.DataFrame(metrics).sort_values("RMSE"),deployed

def feature_row(date,hour,temp,humidity,wind,visibility,dew,solar,rain,snow,season,holiday,functioning):
    d=pd.Timestamp(date); return pd.DataFrame([{
      "hour":hour,"temperature_c":temp,"humidity_pct":humidity,"wind_speed_ms":wind,"visibility_10m":visibility,"dew_point_temperature_c":dew,"solar_radiation_mj_m2":solar,"rainfall_mm":rain,"snowfall_cm":snow,"season":season,"holiday":holiday,"functioning_day":functioning,"year":d.year,"month":d.month,"day_of_week":d.dayofweek,"is_weekend":int(d.dayofweek>=5),"hour_sin":np.sin(2*np.pi*hour/24),"hour_cos":np.cos(2*np.pi*hour/24)}])

df=load_data(); metrics,models=train_models(df)
st.title("Seoul Bike Sharing Demand Predictor")
st.caption("A deployable regression app using the UCI Seoul Bike Sharing Demand dataset.")
tab1,tab2,tab3=st.tabs(["Predict demand","Model comparison","Dataset & EDA"])
with tab1:
    st.subheader("Enter conditions")
    c1,c2,c3=st.columns(3)
    with c1:
        date=st.date_input("Date",value=pd.Timestamp("2018-06-15")); hour=st.slider("Hour",0,23,18); season=st.selectbox("Season",["Spring","Summer","Autumn","Winter"],index=1); functioning=st.selectbox("Functioning day",["Yes","No"]); holiday=st.selectbox("Holiday",["No Holiday","Holiday"])
    with c2:
        temp=st.number_input("Temperature (°C)",value=20.0); humidity=st.number_input("Humidity (%)",0.0,100.0,55.0); wind=st.number_input("Wind speed (m/s)",0.0,50.0,1.5); visibility=st.number_input("Visibility (10m units)",0.0,5000.0,1500.0)
    with c3:
        dew=st.number_input("Dew point temperature (°C)",value=10.0); solar=st.number_input("Solar radiation (MJ/m²)",0.0,10.0,0.5); rain=st.number_input("Rainfall (mm)",0.0,100.0,0.0); snow=st.number_input("Snowfall (cm)",0.0,50.0,0.0)
    chosen=st.selectbox("Model",["Ridge Regression","Lasso Regression","ElasticNet Regression"],index=0)
    if st.button("Predict rented bikes",type="primary"):
        row=feature_row(date,hour,temp,humidity,wind,visibility,dew,solar,rain,snow,season,holiday,functioning); prediction=max(0,float(models[chosen].predict(row)[0])); st.success(f"Estimated hourly demand: {prediction:,.0f} bikes"); st.info("This estimate is a model output, not a guaranteed operational count. Monitor error and retrain as demand patterns change.")
with tab2:
    st.subheader("Chronological 80/20 evaluation")
    st.dataframe(metrics.style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","R²":"{:.3f}"}),use_container_width=True)
    st.bar_chart(metrics.set_index("Model")["RMSE"])
    best=metrics[metrics.Model!="Mean Baseline"].iloc[0]; st.write(f"Best regularized model by RMSE: **{best.Model}**")
with tab3:
    st.subheader("Dataset overview")
    a,b,c=st.columns(3); a.metric("Rows",f"{len(df):,}"); b.metric("Features",f"{len(FEATURES)}"); c.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
    st.write("Average rented bikes by hour")
    st.line_chart(df.groupby("hour")[TARGET].mean())
    st.write("Average rented bikes by season")
    st.bar_chart(df.groupby("season")[TARGET].mean())
    st.download_button("Download cleaned data",df.to_csv(index=False),"seoul_bike_clean.csv","text/csv")
