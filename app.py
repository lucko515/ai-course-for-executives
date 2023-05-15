import numpy as np
import pandas as pd
import webbrowser
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("AICourse for Executives")
    st.sidebar.title("Lessons")
    lesson_m1 = st.sidebar.selectbox("Section 4 lessons", [
        "Lesson 2: 3 Categories of AI", 
        "Lesson 3: A.I. vs Machine Learning vs. Deep Learning", 
        "Lesson 4: How to Use Machine Learning in Your Company",
        "Lesson 4.1: AI/ML Exercise",
        "Lesson 5: Machine Learning Foundations",
        "Lesson 5.1: Customer segmentation example",
        "Lesson 6: Computer Vision", 
        "Lesson 7: Natural Language Processing", 
        "Lesson 8: AI Hardware", 
        "Lesson 9: AutoML and Transfer Learnign", 
        "Lesson 10: AI Role in Different Industries", 
        "Lesson 11: Challenges and Limitations",
        "Lesson 12: Job Positions"])

    lesson_m2 = st.sidebar.selectbox("Section 3 lessons", [
        "Lesson 2: 3 Categories of AI", 
        "Lesson 3: A.I. vs Machine Learning vs. Deep Learning", 
        "Lesson 4: How to Use Machine Learning in Your Company",
        "Lesson 4.1: AI/ML Exercise",
        "Lesson 5: Machine Learning Foundations",
        "Lesson 5.1: Customer segmentation example",
        "Lesson 6: Computer Vision", 
        "Lesson 7: Natural Language Processing", 
        "Lesson 8: AI Hardware", 
        "Lesson 9: AutoML and Transfer Learnign", 
        "Lesson 10: AI Role in Different Industries", 
        "Lesson 11: Challenges and Limitations",
        "Lesson 12: Job Positions"])
    
    if lesson_m1 == "Lesson 2: 3 Categories of AI":
        st.header("Categories of AI")
        st.write("In this lesson we talked about 3 different categories of AI. Here you can find links to read more about all examples talked in the lesson :)")

    elif lesson_m1 == "Lesson 3: A.I. vs Machine Learning vs. Deep Learning":
        st.header("Categories of AI")
        st.write("Here you can find images, and links regarding AI vs ML vs DL")

        # Step 3: Create a dictionary of resources
        resources = {
            "Resource 1": {"name": "Example Website 1", "url": "https://www.example1.com"},
            "Resource 2": {"name": "Example Website 2", "url": "https://www.example2.com"},
            "Resource 3": {"name": "Example Website 3", "url": "https://www.example3.com"},
            # Add more resources here
        }

        # Step 4: Create a function to open URLs in a new tab
        def open_url(url):
            webbrowser.open_new_tab(url)

        # Step 5: Display resources in a structured format
        for key, value in resources.items():
            st.subheader(key)
            st.write(value["name"])
            if st.button(f"Visit {value['name']}"):
                open_url(value["url"])

    elif lesson_m1 == "Lesson 4: How to Use Machine Learning in Your Company":
        st.header("Netflix case-study")
        st.write("Here you can find images, and links regarding AI vs ML vs DL")

        # Step 3: Create a dictionary of resources
        resources = {
            "Resource 1": {"name": "Example Website 1", "url": "https://www.example1.com"},
            "Resource 2": {"name": "Example Website 2", "url": "https://www.example2.com"},
            "Resource 3": {"name": "Example Website 3", "url": "https://www.example3.com"},
            # Add more resources here
        }

        # Step 4: Create a function to open URLs in a new tab
        def open_url(url):
            webbrowser.open_new_tab(url)

        # Step 5: Display resources in a structured format
        for key, value in resources.items():
            st.subheader(key)
            st.write(value["name"])
            if st.button(f"Visit {value['name']}"):
                open_url(value["url"])

    elif lesson_m1 == "Lesson 4.1: AI/ML Exercise":
        st.header("Lesson 1: Web Traffic Analysis")
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

        elif lesson_m1 == "Lesson 7: Natural Language Processing":
        st.header("Lesson 1: Web Traffic Analysis")
        st.write("In this lesson, you will learn how to analyze web traffic data and predict when to increase your website's bandwidth using machine learning.")

        email_text = st.text_area("Write a simple email here:")

        if st.button("Check for Spam"):
            result = 1 #classify_email(email_text)
            st.write(f"Your email is classified as: {result}")

if __name__ == "__main__":
    main()