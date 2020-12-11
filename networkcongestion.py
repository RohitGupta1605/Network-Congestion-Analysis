import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,11
from datetime import datetime
import pymongo
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['database']
cols= db['traffic']
x=cols.find()
l=[]
for i in x:
    del i['_id']
    l.append(dict(i))
l = sorted(l, key=lambda x: datetime.strptime(x['Date'], '%d-%m-%Y'),reverse=True)
dataset = pd.DataFrame(l)
dataset['Date']=pd.to_datetime(dataset['Date'],infer_datetime_format=True)
indexdataset=dataset.set_index(['Date'])
indexdataset.dropna(inplace=True)
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.plot(indexdataset)
rolmean=indexdataset.rolling(12).mean()
rolstd=indexdataset.rolling(12).std()
rolmean.dropna(inplace=True)
rolstd.dropna(inplace=True)
print(rolmean,rolstd)
org=plt.plot(indexdataset,color='blue',label='Original')
mean=plt.plot(rolmean,color='red',label='Rolling mean')
std=plt.plot(rolstd,color='black',label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling mean and standard deviation')
plt.show(block=False)
from statsmodels.tsa.stattools import adfuller
print('results of dickey fuller test: ')
dftest=adfuller(indexdataset['Traffic'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test statistics','p-value','lags used','no. of observations used'])
for key,value in dftest[4].items():
    dfoutput['Critical values (%s)'%key]=value

print(dfoutput)
indexdataset_logscale=np.log(indexdataset)
indexdataset_logscale.dropna(inplace=True)
plt.plot(indexdataset_logscale)
def test_stationary(timeseries):
    movingaverage=timeseries.rolling(window=12).mean()
    movingstd=timeseries.rolling(window=12).std()
    orig=plt.plot(timeseries,color='blue',label='original')
    mean=plt.plot(movingaverage,color='red',label='rolling mean')
    std=plt.plot(movingstd,color='black',label='rolling std')
    plt.legend(loc='best')
    plt.title('rolling mean & standard deviation')
    plt.show(block=False) 
    print('results of dickey fuller test: ')
    dftest=adfuller(timeseries['Traffic'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test statistics','p-value','lags used','no. of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical values (%s)'%key]=value
    print(dfoutput)
datasetlogshifting=indexdataset_logscale-indexdataset_logscale.shift()
plt.plot(datasetlogshifting)
datasetlogshifting.dropna(inplace=True)
test_stationary(datasetlogshifting)
from statsmodels.tsa.arima_model import ARIMA
datasetlogshifting.dropna(inplace=True)
indexdataset_logscale.dropna(inplace=True)
model=ARIMA(indexdataset_logscale,order=(2,1,1))
results_AR=model.fit(disp=-1)
plt.plot(datasetlogshifting)
actual=datasetlogshifting.max()*indexdataset.max()
predicted=results_AR.fittedvalues.max()*indexdataset.max()
print(actual-predicted)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('rss: %4f '% sum((results_AR.fittedvalues-datasetlogshifting['Traffic'])**2))
print('Plotting AR model')
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(datasetlogshifting, nlags=20)
lag_pacf = pacf(datasetlogshifting, nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogshifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogshifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetlogshifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetlogshifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()
prediction_ARIMA_diff=pd.Series(results_AR.fittedvalues,copy=True)
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
prediction_ARIMA_log = pd.Series(indexdataset_logscale['Traffic'].iloc[0],index=indexdataset_logscale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexdataset)
plt.plot(prediction_ARIMA)
results_AR.plot_predict(1,221)