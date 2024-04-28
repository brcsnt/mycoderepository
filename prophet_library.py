# Case 1

from fbprophet import Prophet
import pandas as pd

# Veri setini yükleme
# Örnek olarak, 'example.csv' adında bir dosyadan veri yüklenecektir.
# Bu dosyanın 'date' adında bir tarih sütunu ve 'consumption' adında bir tüketim sütunu olması gerekmektedir.
df = pd.read_csv('example.csv')
df.rename(columns={'date': 'ds', 'consumption': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# Prophet modelinin kurulumu
model = Prophet()

# Modelin eğitilmesi
model.fit(df)

# Gelecek için tarihlerin oluşturulması
future = model.make_future_dataframe(periods=365)  # Örneğin, bir yıl sonrasına kadar tahmin

# Tahminlerin yapılması
forecast = model.predict(future)

# Tahminlerin görselleştirilmesi
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)




# Case 2

# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setinin yüklenmesi
# Veri setinizin 'date' sütununda tarihler ve 'consumption' sütununda kaynak tüketim değerleri olmalı
df = pd.read_csv('data.csv')
df.rename(columns={'date': 'ds', 'consumption': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# Özelliklerin eklendiği kısım
# Tatil, hafta sonu ve resmi tatil flag'lerini Prophet'a uygun şekilde ekleyelim
df['holiday'] = df['holiday_flag']  # 'holiday_flag' sütunu tatil günlerini gösteriyor
df['weekend'] = df['weekend_flag']  # 'weekend_flag' sütunu hafta sonlarını gösteriyor
df['official_holiday'] = df['official_holiday_flag']  # 'official_holiday_flag' sütunu resmi tatilleri gösteriyor

# Tatilleri tanımla
holidays = df[df['holiday'] == 1][['ds']].copy()
holidays['holiday'] = 'holiday'
holidays['lower_window'] = 0
holidays['upper_window'] = 1

# Hafta sonlarını tanımla
weekends = df[df['weekend'] == 1][['ds']].copy()
weekends['holiday'] = 'weekend'
weekends['lower_window'] = 0
weekends['upper_window'] = 1

# Resmi tatilleri tanımla
official_holidays = df[df['official_holiday'] == 1][['ds']].copy()
official_holidays['holiday'] = 'official_holiday'
official_holidays['lower_window'] = 0
official_holidays['upper_window'] = 1

# Tüm tatilleri birleştir
all_holidays = pd.concat([holidays, weekends, official_holidays])

# Modeli kurma
model = Prophet(holidays=all_holidays)

# Modeli eğitme
model.fit(df)

# Gelecek tarihler için DataFrame oluşturma
future = model.make_future_dataframe(periods=60)  # Örneğin, 60 gün sonrasına kadar tahmin

# Tahminlerin yapılması
forecast = model.predict(future)

# Tahmin sonuçlarını görselleştirme
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# OOT veri seti üzerinde performans ölçümü
# 'oot_data.csv' dosyası Out-of-Time (zaman dışı) veri setinizi içerir
oot_data = pd.read_csv('oot_data.csv')
oot_data['ds'] = pd.to_datetime(oot_data['date'])
oot_data.rename(columns={'consumption': 'y'}, inplace=True)

# OOT veri seti üzerinde tahmin yapılması
oot_forecast = model.predict(oot_data)

# Performans değerlendirmesi
rmse = sqrt(mean_squared_error(oot_data['y'], oot_forecast['yhat']))
print('OOT veri seti üzerindeki RMSE: ', rmse)


# Case 3

# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# Veri setinin yüklenmesi
# Veri setinizin 'date' sütununda tarihler ve 'consumption' sütununda kaynak tüketim değerleri olmalı
df = pd.read_csv('data.csv')
df['ds'] = pd.to_datetime(df['date'])
df.rename(columns={'consumption': 'y'}, inplace=True)

# Bayrakların ve diğer zaman özelliklerinin eklenmesi
# Tatiller, hafta sonları, resmi tatiller, ayın günü, yılın ayı, çeyrek bilgisi, yılın haftası, yarım gün bilgisi
df['holiday'] = df['holiday_flag'].apply(lambda x: 'holiday' if x == 1 else None)
df['weekend'] = df['weekend_flag'].apply(lambda x: 'weekend' if x == 1 else None)
df['official_holiday'] = df['official_holiday_flag'].apply(lambda x: 'official_holiday' if x == 1 else None)
df['day_of_month'] = df['ds'].dt.day
df['month'] = df['ds'].dt.month
df['quarter'] = df['ds'].dt.quarter
df['week_of_year'] = df['ds'].dt.isocalendar().week
df['half_day'] = df['half_day_flag'].apply(lambda x: 'half_day' if x == 1 else None)

# Tatil bilgilerini Prophet'a uygun şekilde hazırlama
holiday_df = pd.DataFrame({
    'holiday': 'holiday',
    'ds': df[df['holiday_flag'] == 1]['ds'],
    'lower_window': 0,
    'upper_window': 1,
})

# Modeli oluşturma ve bayramları eklemek
model = Prophet(holidays=holiday_df, daily_seasonality=True)
model.add_country_holidays(country_name='US')  # Eğer veri setiniz başka bir ülke ise burayı değiştirin

# Özel günlerin ve diğer özelliklerin modelleme için dönüşümünü yapmak
# Burada add_regressor kullanarak ek özellikler ekliyoruz
model.add_regressor('day_of_month')
model.add_regressor('month')
model.add_regressor('quarter')
model.add_regressor('week_of_year')
if 'half_day' in df.columns:
    model.add_regressor('half_day')

# Modeli eğitme
model.fit(df)

# Gelecek tarihler için DataFrame oluşturma ve özelliklerin doldurulması
future = model.make_future_dataframe(periods=60)  # Örneğin, 60 gün sonrasına kadar tahmin
future['day_of_month'] = future['ds'].dt.day
future['month'] = future['ds'].dt.month
future['quarter'] = future['ds'].dt.quarter
future['week_of_year'] = future['ds'].dt.isocalendar().week
future['half_day'] = 0  # Gelecekteki yarım gün tahmini yok varsayalım

# Tahminlerin yapılması
forecast = model.predict(future)

# Tahmin sonuçlarını görselleştirme
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

# OOT veri seti üzerinde performans ölçümü
oot_data = pd.read_csv('oot_data.csv')
oot_data['ds'] = pd.to_datetime(oot_data['date'])
oot_data.rename(columns={'consumption': 'y'}, inplace=True)
oot_forecast = model.predict(oot_data)

# Performans değerlendirmesi
rmse = sqrt(mean_squared_error(oot_data['y'], oot_forecast['yhat']))
print('OOT veri seti üzerindeki RMSE: ', rmse)



import plotly.graph_objs as go
import pandas as pd

# OOT veri seti ve tahminlerin yüklenmesi
oot_data = pd.read_csv('oot_data.csv')
oot_data['ds'] = pd.to_datetime(oot_data['date'])
oot_data.rename(columns={'consumption': 'y'}, inplace=True)

# Modelin tahminlerini yapan kısımı tekrar kullanalım
oot_forecast = model.predict(oot_data)

# Plotly ile görselleştirme
trace1 = go.Scatter(
    x=oot_data['ds'],
    y=oot_data['y'],
    mode='markers',
    name='Gerçek Değerler'
)

trace2 = go.Scatter(
    x=oot_forecast['ds'],
    y=oot_forecast['yhat'],
    mode='lines',
    name='Tahminler'
)

trace3 = go.Scatter(
    x=oot_forecast['ds'],
    y=oot_forecast['yhat_upper'],
    fill=None,
    mode='lines',
    line={'width': 0},
    showlegend=False
)

trace4 = go.Scatter(
    x=oot_forecast['ds'],
    y=oot_forecast['yhat_lower'],
    fill='tonexty',  # bu satır, üst sınır ile alt sınır arasını doldurur
    mode='lines',
    line={'width': 0},
    showlegend=False
)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(
    title='Model Tahminleri ve Gerçek Veri Değerleri',
    xaxis=dict(title='Tarih'),
    yaxis=dict(title='Kaynak Tüketimi')
)

fig = go.Figure(data=data, layout=layout)
fig.show()
