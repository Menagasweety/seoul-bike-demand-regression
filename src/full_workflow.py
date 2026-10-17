"""Full data-science workflow for Seoul Bike Sharing Demand.

Stages: data audit, processing, EDA exports, mean baseline, Ridge/Lasso/ElasticNet,
chronological comparison, and final deployment pipeline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'/'seoul_bike_clean.csv'
REPORTS=ROOT/'reports'
TARGET='rented_bike_count'
FEATURES=['hour','temperature_c','humidity_pct','wind_speed_ms','visibility_10m','dew_point_temperature_c','solar_radiation_mj_m2','rainfall_mm','snowfall_cm','season','holiday','functioning_day','year','month','day_of_week','is_weekend','hour_sin','hour_cos']
CAT=['season','holiday','functioning_day']; NUM=[c for c in FEATURES if c not in CAT]

def make_pipeline(model):
    prep=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore'),CAT)])
    return Pipeline([('preprocess',prep),('model',model)])

def main():
    df=pd.read_csv(DATA,parse_dates=['date']).sort_values(['date','hour']).reset_index(drop=True)
    audit=pd.DataFrame({'column':df.columns,'dtype':[str(df[c].dtype) for c in df.columns],'missing_count':[int(df[c].isna().sum()) for c in df.columns],'unique_count':[int(df[c].nunique()) for c in df.columns]})
    audit.to_csv(REPORTS/'data_audit.csv',index=False)
    split=int(len(df)*0.8); X_train,X_test=df.loc[:split-1,FEATURES],df.loc[split:,FEATURES]; y_train,y_test=df.loc[:split-1,TARGET],df.loc[split:,TARGET]
    model_defs={'Ridge Regression':Ridge(alpha=10.0),'Lasso Regression':Lasso(alpha=0.05,max_iter=20000,random_state=42),'ElasticNet Regression':ElasticNet(alpha=0.05,l1_ratio=0.5,max_iter=20000,random_state=42)}
    rows=[{'model':'Mean Baseline','MAE':mean_absolute_error(y_test,np.full(len(y_test),y_train.mean())),'RMSE':mean_squared_error(y_test,np.full(len(y_test),y_train.mean()))**0.5,'R2':r2_score(y_test,np.full(len(y_test),y_train.mean()))}]
    for name,model in model_defs.items():
        pipe=make_pipeline(model); pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
        rows.append({'model':name,'MAE':mean_absolute_error(y_test,pred),'RMSE':mean_squared_error(y_test,pred)**0.5,'R2':r2_score(y_test,pred)})
    metrics=pd.DataFrame(rows).sort_values('RMSE'); metrics.to_csv(REPORTS/'regularized_model_metrics.csv',index=False); print(metrics.to_string(index=False)); print('Recommended:',metrics[metrics.model!='Mean Baseline'].iloc[0].model)

if __name__=='__main__': main()
