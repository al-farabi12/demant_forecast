import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Multi-Outlet Time Series Forecasting")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # User selects columns
    outlet_column = st.selectbox("Select the outlet column:", df.columns)
    date_column = st.selectbox("Select the date column:", df.columns)
    value_column = st.selectbox("Select the value column to forecast:", df.columns)

    forecast_period = st.number_input("Number of days to forecast:", min_value=1, max_value=365, value=30)

    if st.button("Run Forecasts for Each Outlet"):
        outlets = df[outlet_column].unique()
        st.write(f"Found {len(outlets)} outlets: {outlets}")

        for outlet in outlets:
            st.subheader(f"Forecast for {outlet}")

            outlet_df = df[df[outlet_column] == outlet][[date_column, value_column]]
            outlet_df = outlet_df.rename(columns={date_column: 'ds', value_column: 'y'})

            # Check for sufficient data
            if len(outlet_df) < 10:
                st.warning(f"Not enough data to forecast for {outlet}. Skipping.")
                continue

            model = Prophet()
            model.fit(outlet_df)

            future = model.make_future_dataframe(periods=forecast_period)
            forecast = model.predict(future)

            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)