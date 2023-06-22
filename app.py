import numpy as np
import pandas as pd
import webbrowser
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

 # Step 4: Create a function to open URLs in a new tab


def main():
    st.title("AICourse for Executives")
    st.sidebar.title("Lessons")
    lesson_m1 = st.sidebar.selectbox("Section 2 exercises", [
        "Lesson 4.1: AI/ML Exercise", 
        "Lesson 5.1: Customer segmentation example"])

    if lesson_m1 == "Lesson 4.1: AI/ML Exercise":
        st.header("Lesson 4.1: Web Traffic Analysis")
        st.write("In this lesson, you will learn how to analyze web traffic data and predict when to increase your website's bandwidth using machine learning.")

        st.subheader("Upload your CSV file")
        file = st.file_uploader("Upload the web_traffic.csv file provided in the course material for this lesson", type=["csv"])

        if file is not None:
            data = pd.read_csv(file)
            st.write("Here's a preview of the data:")
            st.write(data.head())

            st.subheader("Perform Prediction and Anomaly Detection")
            if st.button("Run Prediction"):
                # Splitting the data into training and testing sets
                data['date_hour'] = pd.to_datetime(data['date_hour'])
                data['date_hour'] = data['date_hour'].apply(lambda x: x.timestamp())

                X = data[['date_hour']].values
                y = data['web_traffic'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Trainingthec linearo regressionn modeltinu
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predicting using the trained model
                y_pred = model.predict(X_test)

                # Calculating metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Displaying the results
                # st.write("Mean Squared Error:", mse)
                # st.write("R-squared Value:", r2)

                # Detecting anomalies
                residuals = y_test - y_pred
                threshold = 1.5 * residuals.std()
                anomalies = residuals[abs(residuals) > threshold]

                anomaly_indices = np.where(abs(residuals) > threshold)[0]
                anomaly_dates = data.loc[anomaly_indices, 'date_hour'].apply(lambda x: datetime.fromtimestamp(x))
                anomaly_data = pd.DataFrame({"Date and Hour": anomaly_dates.dt.strftime('%Y-%m-%d %H:%M:%S'), "Anomaly Value": anomalies})
                st.subheader("Anomalies Detected")
                st.write(f"Found {len(anomalies)} anomalies.")
                st.write("Anomaly Dates and Hours with Values:")
                st.write(anomaly_data)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, label='Regular Data', alpha=0.7)
                ax.scatter(X_test[anomaly_indices], y_test[anomaly_indices], c='red', label='Anomalies', alpha=0.7)
                ax.set_xlabel('Timestamp (seconds)')
                ax.set_ylabel('Web Traffic')
                ax.legend()
                
                st.subheader("Web Traffic Anomalies Visualization")
                st.pyplot(fig)

    elif lesson_m1 == "Lesson 5.1: Customer segmentation example":
        st.header("Lesson 5.1: Customer segmentation example")
        st.write("In this lesson, we will use user's data from a shopping mall and cluster them so that we can help marketing team position them in a better way.")

        st.subheader("Upload your CSV file")
        file = st.file_uploader("Upload the mall_users.csv (file you downloaded from Kaggle) or donwload from the course materials for this lesson", type=["csv"])

        if file is not None:
            data = pd.read_csv(file)
            st.write("Here's a preview of the data:")
            st.write(data.head())

            st.subheader("Perform Clustering on Mall Users! ")
            if st.button("Run Clustering"):
                customers_data = data.drop('CustomerID', axis=1)
                # Splitting the data into training and testing sets
                encode = LabelEncoder()
                encoded_sex = encode.fit_transform(customers_data['Gender'])
                customers_data['Gender'] = encoded_sex
                
                pca_reducer = PCA(n_components=2)
                reduced_data = pca_reducer.fit_transform(customers_data)
                km = KMeans(n_clusters=5)
                cluster = km.fit(reduced_data)

                # Predicting using the trained model
                y_pred = km.predict(reduced_data)
                centroids = km.cluster_centers_
                plt.figure(figsize=(12, 6))
                plt.scatter(reduced_data[y_pred == 0, 0], reduced_data[y_pred == 0, 1], c='r', label='Cluster One')
                plt.scatter(reduced_data[y_pred == 1, 0], reduced_data[y_pred == 1, 1], c='b', label='Cluster two')
                plt.scatter(reduced_data[y_pred == 2, 0], reduced_data[y_pred == 2, 1], c='g', label='Cluster three')
                plt.scatter(reduced_data[y_pred == 3, 0], reduced_data[y_pred == 3, 1], c='y', label='Cluster four')
                plt.scatter(reduced_data[y_pred == 4, 0], reduced_data[y_pred == 4, 1], color='orange', label='Cluster five')
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='black', label='Centroids')
                plt.title("Custom KMeans results")
                plt.legend()
                
                st.subheader("Segmented customers")
                st.pyplot(plt)

                st.subheader("Customer segment 1")
                st.write(data[y_pred==0])

                st.subheader("Customer segment 2")
                st.write(data[y_pred==1])

                st.subheader("Customer segment 3")
                st.write(data[y_pred==2])

                st.subheader("Customer segment 4")
                st.write(data[y_pred==3])

                st.subheader("Customer segment 5")
                st.write(data[y_pred==4])


if __name__ == "__main__":
    main()