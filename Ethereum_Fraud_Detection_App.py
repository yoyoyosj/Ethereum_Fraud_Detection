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
        # Interpret the prediction
        if prediction[0] == 0:
            return 'Non-fraud', 'success'
        else:
            return 'Fraud', 'error'
    except Exception as e:
        return str(e)

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Ethereum Fraud Detection ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    mode = st.radio("Select Mode", ("Batch Prediction", "Real-time Prediction"))

    if mode == "Batch Prediction":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            
            if input_df.isnull().values.any():
                st.warning("Error occurred. Please check your file.")
                return

            input_data = input_df.iloc[:, 2:].values.tolist()  # Exclude the first two columns (address and flag)
            true_labels = input_df['flag'].values
            
            if st.button("Predict"):
                with st.spinner("Processing..."):
                    predictions = fraud_prediction(input_data)
                    predicted_labels = [0 if pred == 'Non-fraud' else 1 for pred in predictions]
                    
                    # Debugging output
                    st.write("Length of true_labels:", len(true_labels))
                    st.write("Length of predicted_labels:", len(predicted_labels))
                    
                    # Calculate evaluation metrics only if lengths match
                    if len(true_labels) == len(predicted_labels):
                        accuracy = accuracy_score(true_labels, predicted_labels)
                        precision = precision_score(true_labels, predicted_labels)
                        recall = recall_score(true_labels, predicted_labels)
                        f1 = f1_score(true_labels, predicted_labels)
                        st.success("Predictions Complete!")
                        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
                        st.write(f"**Precision:** {precision:.2f}")
                        st.write(f"**Recall:** {recall:.2f}")
                        st.write(f"**F1 Score:** {f1:.2f}")
                    else:
                        st.error("Lengths of true_labels and predicted_labels do not match.")
                        st.stop()
                    
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

    elif mode == "Real-time Prediction":
        # Inputs for real-time prediction
        # You can add appropriate input widgets here

        result = ""
        if st.button("Predict"):
            # Collect input data here
            # Example: minTimeBetweenSentTnx = st.text_input("minTimeBetweenSentTnx", "Type Here")
            
            # Call fraud_prediction function with input data
            result, prediction_status = fraud_prediction([[minTimeBetweenSentTnx, maxTimeBetweenSentTnx, avgTimeBetweenSentTnx, minTimeBetweenRecTnx, maxTimeBetweenRecTnx, avgTimeBetweenRecTnx, lifetime, sentTransactions, receivedTransactions, createdContracts, numUniqSentAddress, numUniqRecAddress, minValSent, maxValSent, avgValSent, minValReceived, maxValReceived, avgValReceived, totalTransactions, totalEtherSent, totalEtherReceived, totalEtherSentContracts, totalEtherBalance, activityDays, dailyMax, ratioRecSent, ratioSentTotal, ratioRecTotal, giniSent, giniRec, txFreq, stdBalanceEth]])
            if prediction_status == 'success':
                st.success(result)
            else:
                st.error(result)

if __name__ == '__main__':
    main()
