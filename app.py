from os import write
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import xgboost as xg
import plotly_express as px





st.title('House Price Predictor')

st.write('Below are the descriptions of the features')

st.write('waterfront - Variable for whether the property was overlooking the waterfront or not')

st.write('view - An index from 0 to 4 of how good the view of the property was')

st.write("condition - An index from 1 to 5 on the condition of the property")

st.write("grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design")


st.sidebar.header('House Features')
image = Image.open('house.jpg')
st.image(image, '')

dataset = pd.read_csv("House data final.csv")

# Function
def user_report ():
    bedrooms = st.sidebar.slider('bedroom',1,20,1,key=1)
    bathrooms = st.sidebar.slider('bathroom', 1,20,1,key=2)
    sqft_living = st.sidebar.slider('sqft living', 200,20000,200,key=3)
    sqft_lot = st.sidebar.slider('sqft lot',200,20000,200,key=4 )
    floors = st.sidebar.slider('floor',1,10,1)
    waterfront = st.sidebar.slider('Index for waterfront',0,1,0,key=5)
    view = st.sidebar.slider('Index for view',1,5,1,key=6)
    condition = st.sidebar.slider('Index for condition',1,5,1,key=7)
    grade = st.sidebar.slider('index for grade', 1,12,1,key=8)
    sqft_above = st.sidebar.slider('sqft above', 100,10000,100,key=9)
    sqft_basement = st.sidebar.slider('sqft basement', 100,20000,100,key=10)
    yr_built = st.sidebar.slider('Year built', 1800,2021,1800,key=11)
    yr_renovated = st.sidebar.slider('Year renovated', 0,2021,0,key=12)
    
    
    
    user_report_data = {
        'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft living': sqft_living,
            'sqft lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'grade': grade,
            'sqft above': sqft_above,
            'sqft basement': sqft_basement,
            'year built': yr_built,
            'year renovated': yr_renovated,
            

         }
    
    features = pd.DataFrame(user_report_data, index=[1])
    return features

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


base_model = xg.XGBRegressor()
base_model.fit(X_train,y_train)

model= base_model
user_data = user_report()
st.header('House price prediction')
st.write(user_data)


salary = model.predict(user_data)
st.subheader('House Price is')
st.subheader('$'+str(np.round(salary[0], 2)))






global df
#configuration

