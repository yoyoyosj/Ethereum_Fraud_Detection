import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Load the saved model and preprocessing objects
loaded_model = pickle.load(open("trained_model.sav", "rb"))
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Function for prediction
def fraud_prediction(input_data):
    try:
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(len(input_data_as_numpy_array), -1)
        input_data_scaled = scaler.transform(input_data_reshaped)
        input_data_pca = pca.transform(input_data_scaled)
        prediction = loaded_model.predict(input_data_pca)
        result = ['Non-fraud' if pred == 0 else 'Fraud' for pred in prediction]
        return result
    except Exception as e:
        return str(e)

def main():
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Ethereum Fraud Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Path to image file or a URL
    image_path = "img/picc.jpg"  # Replace with your image file path or URL

    # Display the image
    st.image(image_path, use_column_width=True)
    
    mode = st.radio("Select Mode", ("Batch Prediction", "Manual Input"))

    if mode == "Batch Prediction":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            
            if input_df.isnull().values.any():
                st.warning("Error occured. Please check your file.")
                return

            input_data = input_df.iloc[:, 2:].values.tolist()  # Exclude the first two columns (address and flag)
            true_labels = input_df['flag'].values
            
            if st.button("Predict"):
                with st.spinner("Processing..."):
                    predictions = fraud_prediction(input_data)
                    predicted_labels = [0 if pred == 'Non-fraud' else 1 for pred in predictions]
                    
                    accuracy = accuracy_score(true_labels, predicted_labels)
                    precision = precision_score(true_labels, predicted_labels)
                    recall = recall_score(true_labels, predicted_labels)
                    f1 = f1_score(true_labels, predicted_labels)
                
                st.success("Predictions Complete!")
                st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")
                
                input_df['Prediction'] = predictions
                st.write(input_df)

                fig = px.histogram(input_df, x='Prediction', title='Fraud vs Non-Fraud Transactions')
                st.plotly_chart(fig)

                st.download_button(
                    label="Download Predictions Result",
                    data=input_df.to_csv().encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )

    elif mode == "Manual Input":
        minTimeBetweenSentTnx = st.text_input("Minimum time between sent transactions", "Type Here")
        maxTimeBetweenSentTnx = st.text_input("Maximum time between sent transactions", "Type Here")
        avgTimeBetweenSentTnx = st.text_input("Average time between sent transactions", "Type Here")
        minTimeBetweenRecTnx = st.text_input("Minimum time between received transactions", "Type Here")
        maxTimeBetweenRecTnx = st.text_input("Maximum time between received transactions", "Type Here")
        avgTimeBetweenRecTnx = st.text_input("Average time between received transactions", "Type Here")
        lifetime = st.text_input("Lifetime of the transactions in days", "Type Here")
        sentTransactions = st.text_input("Number of sent transactions", "Type Here")
        receivedTransactions = st.text_input("Number of received transactions", "Type Here")
        createdContracts = st.text_input("Number of created contracts", "Type Here")
        numUniqSentAddress = st.text_input("Number of unique addresses the address sent transactions to", "Type Here")
        numUniqRecAddress = st.text_input("Number of unique addresses from which the address received transactions", "Type Here")
        minValSent = st.text_input("Minimum value of sent transactions", "Type Here")
        maxValSent = st.text_input("Maximum value of sent transactions", "Type Here")
        avgValSent = st.text_input("Average value of sent transactions", "Type Here")
        minValReceived = st.text_input("Minimum value of received transactions", "Type Here")
        maxValReceived = st.text_input("Maximum value of received transactions", "Type Here")
        avgValReceived = st.text_input("Average value of received transactions", "Type Here")
        totalTransactions = st.text_input("Total number of transactions", "Type Here")
        totalEtherSent = st.text_input("Total amount of Ether sent", "Type Here")
        totalEtherReceived = st.text_input("totalEtherReceived", "Type Here")
        totalEtherSentContracts = st.text_input("Total amount of Ether received", "Type Here")
        totalEtherBalance = st.text_input("Total Ether balance", "Type Here")
        activityDays = st.text_input("Number of days the transaction was active", "Type Here")
        dailyMax = st.text_input("Maximum transactions in a single day", "Type Here")
        ratioRecSent = st.text_input("Ratio of received to sent transactions", "Type Here")
        ratioSentTotal = st.text_input("Ratio of sent transactions to total transactions", "Type Here")
        ratioRecTotal = st.text_input("Ratio of received transactions to total transactions", "Type Here")
        giniSent = st.text_input("Gini coefficient for sent transactions", "Type Here")
        giniRec = st.text_input("Gini coefficient for received transactions", "Type Here")
        txFreq = st.text_input("Transaction frequency", "Type Here")
        stdBalanceEth = st.text_input("Standard deviation of Ether balance", "Type Here")

        result = ""
        if st.button("Predict"):
            result = fraud_prediction([[minTimeBetweenSentTnx, maxTimeBetweenSentTnx, avgTimeBetweenSentTnx, minTimeBetweenRecTnx, maxTimeBetweenRecTnx, avgTimeBetweenRecTnx, lifetime, sentTransactions, receivedTransactions, createdContracts, numUniqSentAddress, numUniqRecAddress, minValSent, maxValSent, avgValSent, minValReceived, maxValReceived, avgValReceived, totalTransactions, totalEtherSent, totalEtherReceived, totalEtherSentContracts, totalEtherBalance, activityDays, dailyMax, ratioRecSent, ratioSentTotal, ratioRecTotal, giniSent, giniRec, txFreq, stdBalanceEth]])
            if 'Fraud' in result:
                st.error("Fraud detected!")
            else:
                st.success("No Fraud detected.")


if __name__ == '__main__':
    main()
