import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd
import requests
import pickle

# Loading sklearn model:

with open('models//LinearRegression.pkl', 'rb') as f:
    model = pickle.load(f)


# Geting weather data from API:

api = 'https://api.openweathermap.org/data/2.5/forecast?lat=26.797382&lon=53.350536&units=metric&appid=b71a868e617bf0f03e0a857d1dd6e7df'

data = requests.get(api)
data = data.json()
data = data['list']

# Parsing weather data:

time = []
t = []  # temperature
p = []  # pressure
h = []  # humidity

for item in data:
    time.append(item['dt'])
    
    weather = item['main']
    t.append(weather['temp'])
    p.append(weather['pressure'])
    h.append(weather['humidity'])
    
time = [datetime.fromtimestamp(i) for i in time]

# Predicting power generation:

data = list(zip(t, p, h))
data = pd.DataFrame(data, columns=['temp', 'amb_p', 'rel_h'])
predicted_power = model.predict(data)

# Ploting weather data:

fig0, ax = plt.subplots(3, figsize=(15, 15), sharex=True)
titles = ['temp', 'pressure', 'rel humidity']
labels = ['C', 'Kpa', '%']
params = [t, p, h]

for ax, param, title, label in zip(ax.ravel(), params, titles, labels):
    ax.plot(time, param, color='indigo')
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(label, fontsize=20)
    ax.grid()

fig0.tight_layout(pad=2)

# Ploting predicted power:

fig1, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, predicted_power, color='crimson')
ax.set_title('Predicted power production', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('MW/hr', fontsize=20)
ax.grid()

# Loading logo:

img = Image.open('data//logo.jpg')

# Streamlit:

st.title('Lavan oil refinery company')
st.image(img, width=200)
st.write('Forecasting power generation of CCPP based on forecasted weather data:')
st.header('Forecasted weather of the next 5 days:')
st.pyplot(fig0)

st.header('predicted power production:')
st.pyplot(fig1)
st.info('Developed by Mahdi Hajebi')
st.info('@mertomax')
