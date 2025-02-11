import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
from PIL import Image

model = pk.load(open('rf_model.pkl','rb'))

st.title('VELOCITY')

# Load car data
cars_data = pd.read_csv('Cardetails.csv')

# Function to get brand name from car name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Apply get_brand_name function to 'name' column
cars_data['brand'] = cars_data['name'].apply(get_brand_name)

# Sidebar for other details
st.sidebar.header('Input Features')

# Car brand
name = st.sidebar.selectbox('Select Car Brand', cars_data['brand'].unique())

# Car Manufactured Year
year = st.sidebar.slider('Car Manufactured Year', 1994, 2024)

# No of kms Driven
km_driven = st.sidebar.slider('No of kms Driven', 11, 200000)

# Fuel type
fuel = st.sidebar.selectbox('Fuel type', cars_data['fuel'].unique())

# Seller type
seller_type = st.sidebar.selectbox('Seller Type', cars_data['seller_type'].unique())

# Transmission type
transmission = st.sidebar.selectbox('Transmission Type', cars_data['transmission'].unique())

# Owner type
owner = st.sidebar.selectbox('Owner Type', cars_data['owner'].unique())

# Car Mileage
mileage = st.sidebar.slider('Car Mileage', 10, 40)

# Engine CC
engine = st.sidebar.slider('Engine CC', 700, 5000)

# Max Power
max_power = st.sidebar.slider('Max Power', 0, 200)

# No of Seats
seats = st.sidebar.slider('No of Seats', 5, 10)

# Image path
image_path = 'car_image.jpg'  # Specify the path to your image file

# Load and display image
image = Image.open('abc.jpg')
st.image(image, caption='Car Image', use_column_width=True, output_format='JPEG')

if st.button("Predict"):
    # Create input dataframe
    input_data_model = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Map categorical variables to numerical values
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'] = input_data_model['name'].apply(get_brand_name)  # Get brand name
    input_data_model['name'].replace(cars_data['brand'].unique(), range(1, len(cars_data['brand'].unique()) + 1), inplace=True)  # Map brand to numerical value

    # Make prediction
    car_price = model.predict(input_data_model)

    # Display prediction
    st.header('Prediction')
    st.write(f'Estimated Car Price: {car_price[0]:,.2f} INR')
