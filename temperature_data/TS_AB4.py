# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:52:48 2025

@author: moreo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import xarray as xr
import cfgrib
from statsmodels.tsa.seasonal import STL
from scipy.stats import kurtosis, skew
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf

import warnings
warnings.filterwarnings("ignore")

# ds = xr.open_dataset(r'D:\workspace_HE\TemRisk\0_rawdata\tmp ts new\1990.grib', engine='cfgrib')
# df = ds.to_dataframe()
# df = df.reset_index()
# df = df[['time','latitude','longitude','t2m']]

# ds = xr.open_dataset(r'D:\workspace_HE\TemRisk\0_rawdata\tmp ts new\1993.grib', engine='cfgrib')
# df2 = ds.to_dataframe()
# df2 = df2.reset_index()
# df2 = df2[['time','latitude','longitude','t2m']]

# def vis_stl():
#     fig, axes = plt.subplots(2, 4, figsize=(12, 8))
#     axes[0][0].plot(ts1['t2m'], label="Original")
#     axes[0][0].legend()
#     axes[0][1].plot(trend1, label="T1")
#     axes[0][1].legend()
#     axes[0][2].plot(seasonal1, label="S1")
#     axes[0][2].legend()
#     axes[0][3].plot(resid1, label="R1")
#     axes[0][3].legend()
    
#     axes[1][0].plot(daily_trend["trend"], label="Original")
#     axes[1][0].legend()
#     axes[1][1].plot(trend2, label="T2")
#     axes[1][1].legend()
#     axes[1][2].plot(seasonal2, label="S2")
#     axes[1][2].legend()
#     axes[1][3].plot(resid2, label="R2")
#     axes[1][3].legend()
    
#     plt.tight_layout()
#     plt.show()

# for x in df['latitude'].unique():
#     for y in df['longitude'].unique():   
#         ts1 = df[(df['latitude']==x) & (df['longitude']==y)]
#         ts2 = df2[(df2['latitude']==x) & (df2['longitude']==y)]
#         ts1 = pd.concat([ts1,ts2])
#         ts1 = ts1.sort_values(by='time')
#         ts1 = ts1.drop_duplicates()
#         ts1 = ts1.reset_index().reset_index()
#         ts1 = ts1[['level_0', 'time', 'latitude', 'longitude', 't2m']]
#         stl_daily = STL(ts1['t2m'], period=24, seasonal=25).fit()
#         trend1, seasonal1, resid1 = stl_daily.trend, stl_daily.seasonal, stl_daily.resid
        
#         daily_trend = pd.concat([ts1['time'],trend1],axis = 1)
#         daily_trend["date"] = daily_trend["time"].dt.date
#         daily_trend = daily_trend.groupby("date")["trend"].mean().reset_index()
#         stl_yearly = STL(daily_trend["trend"], period=365, seasonal=365).fit()
#         trend2, seasonal2, resid2 = stl_yearly.trend, stl_yearly.seasonal, stl_yearly.resid
#         vis_stl()



loc_list = pd.read_csv(r'D:\workspace_HE\TemRisk\0_rawdata\address single.csv')
loc_list = loc_list.astype(float)
loc_name = loc_list.copy()
loc_name['loc'] = loc_name['revlon'].astype(str) + '_' + loc_name['revlat'].astype(str) 

i=0
for loc_tmp in loc_name['loc'].unique():
    tmp_file = r'D:\workspace_HE\TemRisk\0_rawdata\each loc\\'+str(loc_tmp)+'ERA5_Charles.csv'
    tmp_data = pd.read_csv(tmp_file)
    tmp_ts = tmp_data.copy()
    tmp_ts["time"] = pd.to_datetime(tmp_ts["time"])
    tmp_ts["date"] = pd.to_datetime(tmp_ts["time"]).dt.date
    tmp_ts["year"] = pd.to_datetime(tmp_ts["time"]).dt.year
    tmp_ts = tmp_ts[['time',  't2m', 'date', 'year']]
    tmp_attr = year_attr(tmp_ts)
    tmp_attr = tmp_attr.reset_index().melt(id_vars='Year',var_name='Attr_Name',value_name='Attr_Value' ).rename(columns={'index': 'Year'})
    tmp_attr['Loc'] = loc_tmp
    tmp_attr.to_csv(r'D:\workspace_HE\TemRisk\2_result\tem_attr\\'+str(loc_tmp)+'_measure_result.csv')
    i=i+1
    print(i)






ds = xr.open_dataset(r'D:\workspace_HE\Temp Abnormal\2_data\testdata\test_ts.grib', engine='cfgrib')
df2 = ds.to_dataframe()
df2 = df2.reset_index()
df2 = df2[['time','latitude','longitude','t2m']]


test_ts =  df2[(df2['latitude']==df2['latitude'].unique()[0]) & (df2['longitude']==df2['longitude'].unique()[0])]      
test_ts["date"] = test_ts["time"].dt.date
test_ts["year"] = test_ts["time"].dt.year
test_ts_year = test_ts[test_ts["year"]==test_ts["year"].unique()[0]]

#ts_ip = ts_year_norm['resid_d']
def iperiod_attr(ts_ip,label):
    attr_table = pd.DataFrame([[0,0]])
    attr_table.loc[len(attr_table)]=(['mean', ts_ip.mean()])
    attr_table.loc[len(attr_table)]=(['std', ts_ip.std()])
    attr_table.loc[len(attr_table)]=(['kurt', kurtosis(ts_ip)])
    attr_table.loc[len(attr_table)]=(['skew', skew(ts_ip)])
    
    attr_table.columns = ['Attr_Name','Attr_Value']
    
    pacf_values = pd.DataFrame(pacf(ts_ip, nlags=int(np.sqrt(len(ts_ip))), method='ols')).reset_index()    
    first5_pacf = pacf_values[1:6]
    first5_pacf.columns = ['Attr_Name','Attr_Value']
    attr_table = pd.concat([attr_table,first5_pacf],axis = 0)
    attr_table = attr_table.iloc[1:,:]
    attr_table['Attr_Name'] = label + attr_table['Attr_Name'].astype(str)
    return(attr_table)

#iperiod_attr(ts_ip,'R_')

def daily_decomp(test_ts_year):
    
    
    #Step 1. Normalization
    daily_mean = test_ts_year.groupby("date")["t2m"].mean().reset_index()
    daily_std = test_ts_year.groupby("date")["t2m"].std().reset_index()
    ts_year_norm = test_ts_year.copy()
    ts_year_norm = pd.merge(ts_year_norm,daily_mean,on = 'date',how = 'left')
    ts_year_norm = pd.merge(ts_year_norm,daily_std,on = 'date',how = 'left')
    #ts_year_norm.columns = ['time', 'latitude', 'longitude', 't2m', 'date', 'year', 'mean_d', 'std_d']
    ts_year_norm.columns = ['time',  't2m', 'date', 'year', 'mean_d', 'std_d']
    ts_year_norm['t2m_norm'] = (ts_year_norm['t2m'] - ts_year_norm['mean_d'] )/ts_year_norm['std_d']
    
    #Step 2. STL Decomposition
    stl_daily = STL(ts_year_norm['t2m_norm'], period=24, seasonal=25).fit()
    ts_year_norm['trend_d'] = stl_daily.trend
    ts_year_norm['seasonal_d'] = stl_daily.seasonal
    ts_year_norm['resid_d'] = stl_daily.resid
    
    n = len(stl_daily.seasonal)
    fft_result = np.fft.fft(stl_daily.seasonal)  # 傅里叶变换
    frequencies = np.fft.fftfreq(n) 
    amplitude = np.abs(fft_result)     

    sampling_rate = 24  # 每日24个数据点    
    selected_periods = 1/(np.arange(10)+1)
    periods = 1/frequencies/sampling_rate 
    periods = pd.DataFrame(periods).reset_index()
    selected_periods = pd.DataFrame(selected_periods)
    periods[0] = round(periods[0],5)
    selected_periods[0] = round(selected_periods[0],5)
    selected_periods = pd.merge(selected_periods,periods,on = 0, how = 'left')
    selected_amplitudes = amplitude[selected_periods['index']] 
    
    FFT_output = pd.DataFrame([selected_periods[0],selected_amplitudes]).T
    #FFT_output.columns = ["selected_periods","selected_amplitudes"]
    FFT_output.columns = ['Attr_Name','Attr_Value']
    FFT_output['Attr_Name'] = "FFT_" + (FFT_output['Attr_Name'].apply(lambda x: f"{x:.4f}")).astype(str)
    #Step 3. Resid Analysis
    
    Resid_attr = iperiod_attr(ts_year_norm['resid_d'],'Resid_')
    
    #Step 4. Daily Analysis
    
    DM_attr = iperiod_attr(daily_mean['t2m'],'DM_')
    DS_attr = iperiod_attr(daily_std['t2m'],'DS_')
    
    #Gather Attrs
    
    yearly_attr = pd.concat([FFT_output,Resid_attr,DM_attr,DS_attr],axis = 0)
    yearly_attr['Year'] = test_ts_year['year'].unique()[0]
    
    return(yearly_attr)

def year_attr(test_ts):
    year_attr_list = []
    for year_tmp in test_ts["year"].unique():
        print(year_tmp)
        year_attr_list.append(daily_decomp(test_ts[test_ts["year"]==year_tmp]))
    
    year_attr_list = pd.concat(year_attr_list,axis = 0)    
    year_attr_list = pd.pivot_table(year_attr_list, index='Year', columns='Attr_Name', values='Attr_Value', aggfunc='sum')
    return(year_attr_list)
    
year_attr(test_ts)

result_area = []
result_label = []
for x in df2['latitude'].unique()[0:5]:
    for y in df2['longitude'].unique()[0:5]:   
        
        ts2 = df2[(df2['latitude']==x) & (df2['longitude']==y)]
        ts2["date"] = ts2["time"].dt.date
        ts2["year"] = ts2["time"].dt.year
        
        result_area.append(year_attr(ts2))
        result_label.append([x,y])
        print(x,y)

result_area_pool = pd.concat(result_area,axis = 0)

cross_table_long = result_area_pool.reset_index().melt(
    id_vars='Year',  # 保留 year 列
    var_name='Attr_Name',  # 列名转为属性名
    value_name='Attr_Value'  # 值列
).rename(columns={'index': 'Year'})

g = sns.catplot(
    data=cross_table_long,
    x='Year',  # 横轴是 year
    y='Attr_Value',  # 纵轴是值
    col='Attr_Name',  # 每一列作为一个子图
    kind='box',  # 箱型图
    height=4,  # 子图高度
    aspect=2 , # 子图宽高比
    col_wrap = 6,
    sharey = False
)

plt.savefig("boxplot.png", dpi=300, bbox_inches='tight')


#Attr Merge

file_list = os.listdir(r"D:\workspace_HE\TemRisk\2_result\tem_attr")
attr_pool = []
for tmp_file in file_list:
    tmp_data = pd.read_csv(r"D:\workspace_HE\TemRisk\2_result\tem_attr\\" + tmp_file)
    tmp_data = tmp_data[tmp_data.columns[1:]]
    attr_pool.append(tmp_data)
    print(tmp_file)

attr_pool = pd.concat(attr_pool ,axis = 0)
attr_pool.to_csv(r'D:\workspace_HE\TemRisk\2_result\TemAttr.csv')
attr_pool = pd.read_csv(r'D:\workspace_HE\TemRisk\2_result\TemAttr.csv')
attr_pool_ct = pd.pivot(attr_pool,index = ['Year','Loc'], columns = 'Attr_Name',values = 'Attr_Value')
#CLHLS Loading

data_clhls = pd.read_stata(r'D:\workspace_HE\CLHLS\CLHLS 1998-2018 longitudinal\CLHLS 1998-2018 longitudinal\\CLHLS_'+str('1998')+'_2018_longitudinal_dataset_released_version1.dta',convert_categoricals = False)

data_columns = pd.read_csv(r"D:\workspace_HE\TemRisk\1_code\selected_cols.csv")
data_clhls = data_clhls[data_columns['Col_ori']] 
data_clhls_long = data_clhls.melt(id_vars = ['id','a1','v_bthyr','f1_18'],var_name = 'Col_ori',value_name = 'Col_value')
data_clhls_long = pd.merge(data_clhls_long,data_columns,on = 'Col_ori',how = 'left')
data_clhls_long['Age'] = data_clhls_long['Col_Year'].astype(int) - data_clhls_long['v_bthyr'].astype(int)
data_clhls_long = data_clhls_long[['id','Col_Year','a1','Age','f1_18','Col_Val','Col_value']]

data_clhls_ct = pd.pivot(data_clhls_long, index = ['id','Col_Year','a1','Age','f1_18'], columns = 'Col_Val',values = 'Col_value')
data_clhls_ct = data_clhls_ct.reset_index()
data_clhls_ct.columns = ['ID', 'Obs_Year', 'Gender', 'Age', 'Education', 'ADL1', 'ADL2', 'ADL3', 'ADL4','ADL5', 'ADL6', 'Drink', 'Econstate', 'Income', 'Mar', 'SelfH','Smoke']

data_clhls_ct['Gender'] = data_clhls_ct['Gender'].replace({1: 1, 2: 0})
data_clhls_ct['Education'] = data_clhls_ct['Education'].replace({54: np.nan, 65: np.nan, 88: np.nan, 99: np.nan})
data_clhls_ct['Income'] = data_clhls_ct['Income'].replace({99998: 100000, 99999: np.nan})
data_clhls_ct['Econstate'] = data_clhls_ct['Econstate'].replace({1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 8: np.nan, 9: np.nan})

data_clhls_ct['Smoke'] = data_clhls_ct['Smoke'].replace({1: 1, 2: 0, 8: np.nan, 9: np.nan})
data_clhls_ct['Drink'] = data_clhls_ct['Drink'].replace({1: 1, 2: 0, 8: np.nan, 9: np.nan})

adl_columns = ['ADL1', 'ADL2', 'ADL3', 'ADL4','ADL5', 'ADL6']
data_clhls_ct[adl_columns] = data_clhls_ct[adl_columns].replace({1: 0, 2: 1, 3: 1, 8: np.nan, 9: np.nan})
data_clhls_ct['ADL'] = data_clhls_ct[adl_columns].sum(axis=1)
data_clhls_ct['SelfH'] = data_clhls_ct['SelfH'].replace({1: 5, 2: 1, 3: 3, 4: 2, 5: 1, 8: np.nan, 9: np.nan})

data_clhls_ct['Mar'] = data_clhls_ct['Mar'].replace({1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 8: np.nan, 9: np.nan})


#Location Map

data_loc_map = pd.read_csv(r"D:\workspace_HE\TemRisk\0_rawdata\id_location_mapping.csv")
data_loc_map['Loc'] = data_loc_map['revlon'] + '_' + data_loc_map['revlat']
data_loc_map = data_loc_map[['id','Loc']]
data_loc_map.columns = ['ID','Loc']
data_loc_map = data_loc_map.dropna()
data_loc_map['ID'] = data_loc_map['ID'].astype(str)

#Model data
data_clhls_ct['ID'] = data_clhls_ct['ID'].astype(str)
model_data = pd.merge(data_clhls_ct,data_loc_map,on = 'ID',how = 'left')

model_data = model_data[model_data['Loc']!='#VALUE!_#VALUE!']
model_data = model_data[~model_data['Loc'].isna()]
model_data['Obs_Year'] = model_data['Obs_Year'].astype(int)
model_data = pd.merge(model_data,attr_pool_ct,left_on = ['Obs_Year','Loc'],right_on = ['Year','Loc'],how = 'left')


model_data_clean = model_data.drop_duplicates()
model_data_clean.to_csv(r"D:\workspace_HE\TemRisk\2_result\model_for_reg\logi19982018.csv")

import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.preprocessing import StandardScaler

#ADL
model_data_clean = model_data.drop_duplicates()
model_data_clean.replace([0,-8, -6,-7, -9], np.nan, inplace=True)
#model_data_clean = model_data_clean.dropna()
missing_percentage = model_data_clean.isnull().mean() * 100
model_data_clean = model_data_clean[model_data_clean['DM_1'].notna()]
model_data_clean = model_data_clean[model_data_clean['SelfH'].notna()]
y = model_data_clean['SelfH']
X = model_data_clean[["ID",'Obs_Year', 'Gender', 'Age', 'Education', 'Drink', 'Econstate','Mar', 
       'Smoke', 'DM_1', 'DM_2', 'DM_3', 'DM_4', 'DM_5',
       'DM_kurt', 'DM_mean', 'DM_skew', 'DM_std', 'DS_1', 'DS_2', 'DS_3',
       'DS_4', 'DS_5', 'DS_kurt', 'DS_mean', 'DS_skew', 'DS_std', 'FFT_0.1000',
       'FFT_0.1111', 'FFT_0.1250', 'FFT_0.1429', 'FFT_0.1667', 'FFT_0.2000',
       'FFT_0.2500', 'FFT_0.3333', 'FFT_0.5000', 'FFT_1.0000', 'Resid_1',
       'Resid_2', 'Resid_3', 'Resid_4', 'Resid_5', 'Resid_kurt', 'Resid_mean',
       'Resid_skew', 'Resid_std']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X[['DM_1', 'DM_2', 'DM_3', 'DM_4', 'DM_5',
'DM_kurt', 'DM_mean', 'DM_skew', 'DM_std', 'DS_1', 'DS_2', 'DS_3',
'DS_4', 'DS_5', 'DS_kurt', 'DS_mean', 'DS_skew', 'DS_std', 'FFT_0.1000',
'FFT_0.1111', 'FFT_0.1250', 'FFT_0.1429', 'FFT_0.1667', 'FFT_0.2000',
'FFT_0.2500', 'FFT_0.3333', 'FFT_0.5000', 'FFT_1.0000', 'Resid_1',
'Resid_2', 'Resid_3', 'Resid_4', 'Resid_5', 'Resid_kurt', 'Resid_mean',
'Resid_skew', 'Resid_std']] = X_scaled[:,9:]

penalty_weights = np.ones(X_scaled.shape[1])
penalty_weights[0:9] = 0
penalty_weights[-1] = 0
X_data = np.array(X.iloc[:,:])
model = sm.OLS(np.array(y), X_data)
result = model.fit_regularized(alpha=0.1, L1_wt=1.0, start_params=np.zeros(X.shape[1]), penalty_weights=penalty_weights)







