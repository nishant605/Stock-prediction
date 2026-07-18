from cProfile import label
from pathlib import Path
import pandas as pd
import streamlit as st
import pickle
import plotly.express as px
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = Path("models")

df = pd.read_pickle('Copy of df_clean.pkl')

excluded_name = {
    'HINDALC0',    # Old symbol for HINDALCO
    'HEROHONDA',   # Old symbol for HEROMOTOCO
    'HINDLEVER',   # Old symbol for HINDUNILVR
    'INFOSYSTCH',  # Old symbol for INFY
    'TISCO',       # Old symbol for TATASTEEL
    'MUNDRAPORT',  # Old symbol for ADANIPORTS
    'SESAGOA',     # Old symbol for VEDL
    'ZEETELE',     # Old symbol for ZEEL
    'UTIBANK',     # Old symbol for AXISBANK
    'SSLT',        # Old symbol for VEDL (Sterlite)
    'KOTAKMAH',    # Old symbol for KOTAKBANK
    'TELCO',       # Old symbol for TATAMOTORS
    'BHARTI',      # Old symbol for BHARTIARTL
    'UNIPHOS',     # Old symbol (UPL predecessor)
}
all_stocks_raw = sorted(df['Symbol'].unique())
list_of_stocks = [s for s in all_stocks_raw if s not in excluded_name]


st.title('Stock Price Prediction')

features = [
    'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume',
    'Deliverable Volume',
    'return', 'MA10', 'MA50', 'volatility','MACD', 'High_Low_Ratio', 'Close_Open_Ratio'
]

def feature_creation(df, stock_name):
    data = df[df['Symbol'] == stock_name].copy()
    data.sort_values('Date', inplace=True)

    data['return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['volatility'] = data['Close'].rolling(10).std()

    data = data.dropna().reset_index(drop=True)

    ema12 = data['Close'].ewm(span=12, adjust=False).mean() #Calculates the 12-day Exponential Moving Average (EMA) of the closing price.
    ema26 = data['Close'].ewm(span=26, adjust=False).mean() #Means EMA considers 12 previous periods.
    data['MACD'] = ema12 - ema26 #MACD is the difference between short-term EMA and long-term EMA.

    data['High_Low_Ratio'] = data['High'] / (data['Low'] + 1e-10)
    data['Close_Open_Ratio'] = data['Close'] / (data['Open'] + 1e-10)

    data['target'] = data['Close'].pct_change().shift(-1)
    data['current_close'] = data['Close'] #why not use future close price as target?
    #because we want to predict the return, not the price itself. By using percentage change, we can model the relative movement of the stock price,
    #which is more stable and easier for the model to learn compared to absolute price values.

    data = data.dropna().reset_index(drop=True)

    return data

def feature_creation_uploaded(df):
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data['return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['volatility'] = data['Close'].rolling(10).std()
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['High_Low_Ratio'] = (data['High'] / (data['Low'] + 1e-10))
    data['Close_Open_Ratio'] = (data['Close'] / (data['Open'] + 1e-10))
    data['current_close'] = data['Close']
    data['target'] = data['Close'].pct_change().shift(-1)   
    return data

st.subheader('Choose Prediction Method')

prediction_mode = st.radio("Select how you want to provide stock data:", ['Select Stock', 'Upload CSV'],horizontal=True)

if prediction_mode == "Upload CSV":
    st.subheader('Upload Your Stock Data CSV')

    st.info("Please ensure your CSV file contains the following columns: 'Date', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Deliverable Volume'.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error occurred while reading the CSV file: {e}")


        required_columns = ['Date','Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Deliverable Volume']
        missing_columns = [col for col in required_columns if col not in uploaded_df.columns]

        if missing_columns:
            st.error(f"The following required columns are missing in the uploaded CSV: {', '.join(missing_columns)}")
        
        else:
            st.success("File uploaded successfully!")

            uploaded_df['Date'] = pd.to_datetime(uploaded_df['Date'],errors='coerce')  # Convert to datetime, invalid parsing will be set as NaT
            uploaded_df = uploaded_df.dropna(subset=['Date'])  # Drop rows where 'Date' could not be parsed
            if uploaded_df['Date'].isna().any():
                st.error("There are invalid date formats in the 'Date' column. Please ensure all dates are in a valid format (e.g., YYYY-MM-DD).")
            else:
                uploaded_df = uploaded_df.sort_values('Date').reset_index(drop=True)

                st.write(f"Total Rows: {len(uploaded_df)}")

                st.write("Preview of uploaded data:")

                with st.expander("📄 View Uploaded Dataset"):

                    st.dataframe(
                        uploaded_df.tail(10),
                        use_container_width=True
                    )

                if len(uploaded_df) < 200:
                    st.warning("The uploaded data has less than 200 rows. The model may not perform well with limited data.")
                else:
                    st.success("uploaded data has sufficient rows for prediction.")

                    if st.button("train Model & predict"):
                        progress_bar = st.progress(0)
                        status = st.empty()

                        status.text("📂 Reading uploaded data...")
                        progress_bar.progress(10)
                        
                        status.text("⚙️ Creating technical indicators...")
                        full_data = feature_creation_uploaded(uploaded_df)
                        progress_bar.progress(25)

                        latest_data = full_data.dropna(subset=features).iloc[-1:].copy()  # Get the last row of the DataFrame
                        model_data = full_data.dropna(subset=features+['target']).reset_index(drop=True)  # Drop rows with NaN values in features
                        progress_bar.progress(35)

                        status.text("📊 Splitting training and testing data...")
                        train_size = int(len(model_data) * 0.8)

                        train_data = model_data.iloc[:train_size]
                        test_data = model_data.iloc[train_size:]
                        progress_bar.progress(45)

                        # Features
                        status.text("📏 Scaling features...")
                        X_train = train_data[features]
                        X_test = test_data[features]


                        # Target
                        y_train = train_data['target'].values.reshape(-1, 1)
                        y_test = test_data['target'].values.reshape(-1, 1)


                        # Feature Scaling
                        scaler_x = StandardScaler()

                        X_train_scaled = scaler_x.fit_transform(X_train)
                        X_test_scaled = scaler_x.transform(X_test)


                        # Target Scaling
                        scaler_y = StandardScaler()

                        y_train_scaled = scaler_y.fit_transform(y_train)
                        progress_bar.progress(60)


                        # Reshape for GRU
                        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])

                        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

                        status.text("🧠 Building GRU model...")

                        model = Sequential([
                                GRU(64, return_sequences=True, input_shape=(1, len(features))),
                                Dropout(0.2),
                                GRU(32),
                                Dropout(0.2),
                                Dense(16, activation='relu'),
                                Dense(1)
                            ])


                            # Compile Model
                        model.compile(optimizer='adam', loss='mse')
                        progress_bar.progress(70)

                        # Early Stopping
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                        status.text("🚀 Training model...")
                        # Train Model
                        model.fit(
                            X_train_scaled,
                            y_train_scaled,
                            validation_split=0.2,
                            epochs=50,
                            batch_size=32,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        progress_bar.progress(90)

                        # Predict Test Data

                        status.text("📈 Predicting tomorrow's stock price...")
                        y_pred_scaled = model.predict(X_test_scaled, verbose=0)

                        y_pred = scaler_y.inverse_transform(y_pred_scaled)


                        # Convert Return into Price
                        current_close_test = test_data['current_close'].values

                        actual_price = current_close_test * (1 + y_test.flatten())

                        predicted_price_series = current_close_test * (1 + y_pred.flatten())


                        # Predict Tomorrow
                        latest_features = latest_data[features]

                        latest_scaled = scaler_x.transform(latest_features)

                        latest_scaled = latest_scaled.reshape(1, 1, len(features))

                        tomorrow_scaled = model.predict(latest_scaled, verbose=0)

                        tomorrow_return = scaler_y.inverse_transform(tomorrow_scaled)[0][0]


                        # Latest Close Price
                        last_price = latest_data['Close'].iloc[0]


                        # Tomorrow Predicted Price
                        predicted_price = last_price * (1 + tomorrow_return)

                        progress_bar.progress(100)
                        status.success("✅ Training Completed Successfully!")
                        
                        st.success("""
                            ✅ Model trained successfully!

                            The uploaded dataset has been processed and tomorrow's
                            stock price prediction is now available.
                            """)
                        st.divider()

                        st.markdown("## 📊 Historical Stock Price")

                        fig = px.line(
                            uploaded_df,
                            x='Date',
                            y='Close',
                            title='Uploaded Stock Close Price Over Time'
                        )

                        st.plotly_chart(fig, use_container_width=True)


                        # Next Day Prediction
                        st.divider()

                        st.subheader('Predicted Next-Day Close Price')

                        col1, col2,col3= st.columns(3)

                        col1.metric("💰 Latest Close", f"₹{last_price:.2f}")

                        col2.metric(
                            "📈 Tomorrow Prediction",
                            f"₹{predicted_price:.2f}",
                            f"{((predicted_price-last_price)/last_price)*100:.2f}%")

                        col3.metric(
                            "🎯 Predicted Return",
                            f"{tomorrow_return*100:.2f}%"
                        )


                        # Predicted Movement
                        predicted_change = tomorrow_return * 100

                        if tomorrow_return > 0:
                            st.success(f"Predicted Movement: UP ({predicted_change:.2f}%)")

                        elif tomorrow_return < 0:
                            st.warning(f"Predicted Movement: DOWN ({predicted_change:.2f}%)")

                        else:
                            st.info("Predicted Movement: No Change")


                        # Actual vs Predicted
                        st.divider()

                        st.markdown("## 📉 Actual vs Predicted")

                        comparison_df = pd.DataFrame({
                            'Date': test_data['Date'].values,
                            'Actual Close Price': actual_price,
                            'Predicted Close Price': predicted_price_series
                        })

                        result_df = comparison_df.copy()
                        result_df["Tomorrow Predicted Price"] = np.nan
                        result_df.loc[result_df.index[-1], "Tomorrow Predicted Price"] = predicted_price
                        csv = result_df.to_csv(index=False).encode("utf-8")

                        fig2 = px.line(
                            comparison_df,
                            x='Date',
                            y=['Actual Close Price', 'Predicted Close Price'],
                            title='Uploaded Stock Actual vs Predicted Close Price'
                        )

                        st.plotly_chart(fig2, use_container_width=True)


                        # Model Performance
                        st.divider()

                        st.markdown("## 📋 Model Performance")

                        mae = np.mean(np.abs(predicted_price_series - actual_price))

                        mape = np.mean(np.abs((predicted_price_series - actual_price) / actual_price)) * 100

                        direction_accuracy = np.mean(
                            np.sign(predicted_price_series - current_close_test) ==
                            np.sign(actual_price - current_close_test)
                        ) * 100

                        summary_df = pd.DataFrame({
                        "Metric":[
                            "Latest Close Price",
                            "Tomorrow Prediction",
                            "Predicted Return (%)",
                            "MAE",
                            "MAPE",
                            "Direction Accuracy (%)"
                        ],
                        "Value":[
                            last_price,
                            predicted_price,
                            tomorrow_return*100,
                            mae,
                            mape,
                            direction_accuracy
                        ]
                    })
                        
                        summary_csv = summary_df.to_csv(index=False).encode("utf-8")  

                        col1, col2, col3 = st.columns(3)

                        col1.metric("MAE", f"₹{mae:.2f}")

                        col2.metric("MAPE", f"{mape:.2f}%")

                        col3.metric("Direction Accuracy", f"{direction_accuracy:.2f}%")
                        
                        st.divider()
                        st.markdown("## 📥 Download Results")
                        st.download_button(label="📥 Download Prediction Report", data=csv, file_name="prediction_report.csv", mime="text/csv")

                        st.download_button("📄 Download Summary",summary_csv,"prediction_summary.csv","text/csv")
        
else:

    selected_stock = st.selectbox('Select Stock', list_of_stocks)

    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    if st.button('Predict'):
        st.session_state.predict_clicked = True

    if st.session_state.predict_clicked:
        st.divider()

        data = df[df['Symbol'] == selected_stock]
        st.write(f'last 10 days data of {selected_stock}')
        st.dataframe(data.tail(10))

        data = feature_creation(data, selected_stock)
        X = data.drop(['target','Symbol','Date'], axis=1)

        st.divider()

        st.markdown("## 📊 Historical Stock Price")

        fig = px.line(data, x='Date', y='Close', title=f'{selected_stock} Close Price Over Time')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader('Predicted Next-Day Close Price')

        model = load_model(MODEL_DIR / f"gru_model_{selected_stock}.keras")

        scaler_x = pickle.load(open(MODEL_DIR / f"scaler_X_{selected_stock}.pkl", "rb"))
        scaler_y = pickle.load(open(MODEL_DIR / f"scaler_y_{selected_stock}.pkl", "rb"))

        X_scaled = scaler_x.transform(data[features])
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]) #GRU expect this(samples, timesteps, features)
        y_pred_scaled = model.predict(X_scaled)

        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test = scaler_y.inverse_transform(data['target'].values.reshape(-1, 1)) #reshape(-1,1) → converts 1D array to column format.

        last_price = data["current_close"].iloc[-1]
        predicted_price = last_price * (1 + y_pred[-1][0]) #Predicted Price = Current Price × (1 + Predicted Return)

        st.metric(
        "Predicted Close Price for Tomorrow",
        f"₹{predicted_price:.2f}")

        st.divider()

        st.markdown("## 📉 Actual vs Predicted")

        train_size = int(len(data) * 0.8)
        X_test = X_scaled[train_size:]
        y_test = data['target'].values[train_size:]
        dates_test = data['Date'][train_size:]

        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        actual_price = data["current_close"].iloc[train_size:] * (1 + y_test.flatten())
        predicted_price_series = data["current_close"].iloc[train_size:] * (1 + y_pred.flatten()) #Calculates predicted stock price using predicted return

        comparison_df = pd.DataFrame({
            'Date': dates_test,
            'Actual Close Price': actual_price,
            'Predicted Close Price': predicted_price_series
        })

        fig2 = px.line(comparison_df, x='Date', y=['Actual Close Price', 'Predicted Close Price'], 
                    title=f'{selected_stock} Actual vs Predicted Close Price')
        fig2.update_traces(selector=dict(name="Predicted Price"), line=dict(dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("## 📋 Model Performance")

        mae = np.mean(np.abs(predicted_price_series - actual_price))
        mape = np.mean(np.abs((predicted_price_series - actual_price) / actual_price)) * 100 #(predicted - actual) / actual
        direction_accuracy = np.mean((np.sign(np.diff(actual_price)) == np.sign(np.diff(predicted_price_series)))) * 100 #Calculates the percentage of times the model correctly predicted the direction of price movement (up or down) by comparing the signs of the differences between consecutive actual and predicted prices.

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"₹{mae:.2f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("Direction Accuracy", f"{direction_accuracy:.2f}%")

        st.divider()

        st.subheader('Multiple Day Predictions')

        recent_data = data.copy()
        days = st.slider('Select number of days to predict',1,7,1)

        if st.button('Predict Multiple Days'):
            feature_price = []
            last_price = data["current_close"].iloc[-1]
            current_features = X_scaled[-1].reshape(1, 1, -1) #last row of X_scaled

            for day in range(days):
                pred_scaled = model.predict(current_features)
                pred_return = scaler_y.inverse_transform(pred_scaled)[0][0] #predicted return for the day
                next_price = last_price * (1 + pred_return) #Predicted Price = Current
                feature_price.append(next_price)
                
                new_row = recent_data.iloc[-1].copy()
                new_row['Open'] = last_price
                new_row['Close'] = next_price
                new_row['High'] = next_price * 1.005   # small approximation
                new_row['Low'] = next_price * 0.995
                new_row['VWAP'] = next_price
                new_row['return'] = pred_return

                # Append and recalculate rolling features properly
                recent_data = pd.concat([recent_data, new_row.to_frame().T], ignore_index=True)
                recent_data['MA10'] = recent_data['Close'].rolling(10).mean()
                recent_data['MA50'] = recent_data['Close'].rolling(50).mean()
                recent_data['volatility'] = recent_data['Close'].rolling(10).std()
                ema12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
                ema26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
                recent_data['MACD'] = ema12 - ema26
                recent_data['High_Low_Ratio'] = recent_data['High'] / (recent_data['Low'] + 1e-10)
                recent_data['Close_Open_Ratio'] = recent_data['Close'] / (recent_data['Open'] + 1e-10)
                recent_data = recent_data.ffill()

                last_price = next_price

            future_dates = pd.date_range(
            start=data["Date"].iloc[-1],
            periods=days+1,
            freq="B")[1:]

            future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": feature_price})

            st.dataframe(future_df)
            fig_future = px.line(
            future_df,
            x="Date",
            y="Predicted Price",
            title=f"{selected_stock} Next {days} Days Prediction")

            st.plotly_chart(fig_future, use_container_width=True)