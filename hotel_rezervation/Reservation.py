import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from datetime import date
from Hotel_bastan_dogukan import *

ds = pd.read_excel("datasets/HotelEkzel.xlsx")

# Load your trained model
model = joblib.load('lgbm_model.pkl')

# Function to make predictions
def predict_cancellation(data):
    return model.predict(data)
def create_seasonal_features(dataframe):
    dataframe["ARRIVAL_SEASON_Autumn"] = dataframe["ARRIVAL_MONTH"].apply(lambda x: 1 if x in [9, 10, 11] else 0)
    dataframe["ARRIVAL_SEASON_Spring"] = dataframe["ARRIVAL_MONTH"].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    dataframe["ARRIVAL_SEASON_Summer"] = dataframe["ARRIVAL_MONTH"].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    dataframe["ARRIVAL_SEASON_Winter"] = dataframe["ARRIVAL_MONTH"].apply(lambda x: 1 if x in [12, 1, 2] else 0)
    return dataframe

def create_market_features(dataframe):
    dataframe["MARKET_SEGMENT_TYPE_Complementary"] = 0
    dataframe["MARKET_SEGMENT_TYPE_Corporate"] = 0
    dataframe["MARKET_SEGMENT_TYPE_Offline"] = 0
    dataframe["MARKET_SEGMENT_TYPE_Online"] = 1
    dataframe["MARKET_SEGMENT_TYPE_Aviation "] = 0

    return dataframe
# Streamlit UI
def main():
    st.set_page_config(
        page_title="🏝️Make a Reservation Now 🏝"
    )
    st.image('hotel2.jpg')
    st.title("🏝️Make a Reservation Now🏝")
    #st.sidebar.header("HOME")
    #st.sidebar.header("ROOM TYPES")
    #st.sidebar.header("MEAL PLANS")
    #st.sidebar.header("🏝️Get Reservation Now 🏝️")
   # st.sidebar.header("ABOUT")
    #st.sidebar.header("CONTACT")
    # inputs for features
    no_of_adults = st.number_input('Number of Adults', min_value=0,max_value=3, value=0)
    no_of_children = st.number_input('Number of Children', min_value=0,max_value=5, value=0)

    # Selecting room type
    room_type_reserved = st.selectbox('Room Type', ('Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                                           'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))
    # Selection for meal plan
    type_of_meal_plan= st.selectbox('Meal Plan', ('Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'))

    # Date selection for enter and exit dates
    enter_date  = st.date_input("Check-in Date", min_value=datetime.today())
    exit_date = st.date_input("Check-out Date", min_value=enter_date, value=enter_date)


    #Unnecesary features that will be dropped
    Booking_ID="ID879987"

    booking_status=1

    # Seperating day mont year
    arrival_year= enter_date.year
    arrival_month = enter_date.month
    arrival_date = enter_date.day

    # Market Segment Type Online because it is a website :)
    market_segment_type = "Online"

    # it is unnececary and don't have much variable on repeated guests
    repeated_guest = 0
    no_of_previous_cancellations = 0
    no_of_previous_bookings_not_canceled = 0
    # Calculate total stay length
    total_stay_length = (exit_date - enter_date).days

    # Selecting required Parking space
    required_car_parking_spaceB = st.checkbox('Do you need a Park Place', value=False)
    required_car_parking_space = 0
    if (required_car_parking_spaceB):
        required_car_parking_space += 1

    # Selecting number of special requests space
    no_of_special_requests = st.number_input('number of special requests (optinal)', min_value=0, value=0)
    special_requests= list()
    for i in range(no_of_special_requests):
        special_requests.append(st.text_input(f'Special Request {i+1}'))
    dryer_request = st.checkbox('Request a Dryer', value=False)


    #st.write(no_of_special_requests)
    no_of_weekend_nights = 0


    no_of_week_nights = 0


    day_of_week=enter_date.weekday()
    for i in range(total_stay_length):
        if (day_of_week%6 == 0  | day_of_week%6==5):
            no_of_weekend_nights += 1
        else:
            no_of_week_nights += 1


    today = datetime.today().date()
    lead_time = (enter_date - today).days
    avg_price_per_room = 0.00
    avg_price_per_room = ds[(ds['no_of_adults'] == no_of_adults) & (ds['no_of_children'] == no_of_children)
               & (ds['arrival_month'] == arrival_month) &
                            (ds['room_type_reserved'] == room_type_reserved) & (ds['type_of_meal_plan'] == type_of_meal_plan)]['avg_price_per_room'].mean()
    if pd.isna(avg_price_per_room) or avg_price_per_room == "":
        avg_price_per_room = 150.00
        # Collect inputs in a DataFrame
    Date = enter_date

    # Price
    st.subheader(f'Book for just {round(avg_price_per_room * total_stay_length,2)}€ Now')

    input_values = {
        "Booking_ID" : [Booking_ID],
        "no_of_weekend_nights": [no_of_weekend_nights],
        "no_of_week_nights": [no_of_week_nights],
        "arrival_year": [arrival_year],
        "arrival_month": [arrival_month],
        "arrival_date": [arrival_date],
        "market_segment_type": [market_segment_type],
        "repeated_guest": [repeated_guest],
        "no_of_previous_cancellations": [no_of_previous_cancellations],
        "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
        "no_of_adults": [no_of_adults],
        "no_of_children": [no_of_children],
        "room_type_reserved": [room_type_reserved],
        "type_of_meal_plan": [type_of_meal_plan],
        "required_car_parking_space": [required_car_parking_space],
        "no_of_special_requests": [no_of_special_requests],
        "lead_time": [lead_time],
        "avg_price_per_room": [avg_price_per_room],
        "Date": [Date],
        "booking_status" : [booking_status]
    }
    input_data = pd.DataFrame(input_values)

    # Button to make predictions
    if st.button('Predict Cancellation'):
        #Data Preperation
        dk = hotel_data_prep(ds)
        input_data = hotel_data_prep2(input_data, dk)
        cat_cols, num_cols, cat_but_car = grab_col_names(dk)
        num_cols.remove('DATE')
        input_data = create_seasonal_features(input_data)
        input_data = create_market_features(input_data)
        input_data = input_data.drop(["MARKET_SEGMENT_TYPE", "ARRIVAL_SEASON"], axis=1)


        X_scaled = StandardScaler().fit_transform(dk[num_cols])
        input_data[num_cols] = pd.DataFrame(X_scaled, columns=input_data[num_cols].columns)
        X = input_data.drop(["BOOKING_STATUS", "DATE", "BOOKING_ID", "ARRIVAL_YEAR", 'REPEATED_GUEST'], axis=1)

        # Add preprocessing steps here if necessary
        prediction = predict_cancellation(X)
        if (prediction[0] == 1) | (required_car_parking_space == 1) | (lead_time>100) | (no_of_adults/ no_of_children<0.35):
            st.error("The reservation is likely to be canceled.")
        else:
            st.success("The reservation is likely NOT to be canceled.")

if __name__ == "__main__":
    main()
