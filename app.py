#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import streamlit as st

# loading in the model to predict on the data
pickle_in = open('log_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in = open('SVM.pkl', 'rb')
classifier2 = pickle.load(pickle_in)


pickle_in = open('nb.pkl', 'rb')
classifier3 = pickle.load(pickle_in)

classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "SVM", "Naive Bayes"))


def get_classifier():
    if classifier_name == "Logistic Regression":
        return classifier
    elif classifier_name == "SVM":
        return classifier2
    else:
        return classifier3


clf = get_classifier()


# In[ ]:


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(hotel, lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal,
               country, market_segment, is_repeated_guest, distribution_channel, days_in_waiting_list, customer_type,
               adr, required_car_parking_spaces, total_of_special_requests, reservation_status,
               reservation_status_date, Room, net_cancelled, booking_changes, deposit_type, agent, company):
    prediction = classifier.predict(
        [[hotel, lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal,
          country, market_segment, is_repeated_guest, distribution_channel, days_in_waiting_list, customer_type, adr,
          required_car_parking_spaces, total_of_special_requests, reservation_status,
          reservation_status_date, Room, net_cancelled, booking_changes, deposit_type, agent, company]]).astype(
        np.float64)
    print(prediction)
    return prediction


# this is the main function in which we define our webpage 
def main():
    # giving the webpage a title

    st.title("CHURN RATE PREDICTION")

    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;"> DEPLOYMENT </h1>
    </div>
    """

    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    hotel = st.selectbox("WRITE CITY: 0 & RESORT: 1", ('0', '1'))
    lead_time = st.number_input("Provide Lead Time between (0 to 375)")
    stays_in_weekend_nights = st.selectbox("stays in weekend_nights", ('0', '1', '2', '3', '4', '5'))
    stays_in_week_nights = st.selectbox("stays_in_week_nights", ('0', '1', '2', '3', '4', '5', '6'))
    adults = st.selectbox("Number of Adults", ('0', '1', '2', '3'))
    children = st.selectbox("Number of Children", ('0', '1', '2', '3'))
    babies = st.selectbox("Number of Babies", ('0', '1', '2', '3'))
    meal = st.selectbox("Meal", ('0', '1', '2', '3', '4'))
    country = st.number_input("country")
    market_segment = st.selectbox("market_segment", ('0', '1', '2', '3', '4', '5', '6', '7'))
    is_repeated_guest = st.selectbox("Is it a Repeated Guest? NO:0 & YES:1", ('0', '1'))
    distribution_channel = st.selectbox("distribution_channel", ('0', '1', '2', '3', '4'))
    days_in_waiting_list = st.number_input("Days in Waiting list (0 to 365 days)")
    customer_type = st.selectbox("Type of Customer", ('0', '1', '2', '3', '4', '5'))
    adr = st.number_input("adr")
    required_car_parking_spaces = st.selectbox("Required Number of Parking Spaces", ('0', '1', '2', '3'))
    total_of_special_requests = st.selectbox("Total Number of Special Requests", ('0', '1', '2', '3', '4', '5'))
    reservation_status = st.selectbox("Reservation Status", ('1', '0', '2'))
    reservation_status_date = st.number_input("reservation_status_date")
    Room = st.selectbox("Same Room (1 if guest received the same room otherwise 0)", ('0', '1'))
    net_cancelled = st.selectbox("Does Customer Previously Cancelled? 1 : YES & 0 : NO", ('0', '1'))
    booking_changes = st.number_input("booking_changes")
    deposit_type = st.number_input("deposit_type")
    agent = st.number_input("agent")
    company = st.number_input("company")

    result = ""

    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(hotel, lead_time, stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
                            meal,
                            country, market_segment, is_repeated_guest, distribution_channel, days_in_waiting_list,
                            customer_type, adr, required_car_parking_spaces, total_of_special_requests,
                            reservation_status,
                            reservation_status_date, Room, net_cancelled, booking_changes, deposit_type, agent,
                            company)
        if result == 0:
            return st.success('The Customer will CHURN'.format(result))
        else:
            return st.success('The Customer will CONTINUE'.format(result))


if __name__ == '__main__':
    main()
