import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate dates and hours for the whole month
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(hours=x) for x in range(30 * 24)]

# Normal web traffic
np.random.seed(42)
traffic_mean = 2000
traffic_std = 500
web_traffic = np.random.normal(traffic_mean, traffic_std, len(dates))

# Introduce anomalies in the web traffic data
anomalies_count = 20
for _ in range(anomalies_count):
    index = random.randint(0, len(dates) - 1)
    web_traffic[index] *= random.uniform(1.5, 3)

# Create a DataFrame
data = pd.DataFrame({"date_hour": dates, "web_traffic": web_traffic})
data["web_traffic"] = data["web_traffic"].round().astype(int)

# Saveto CSV file
data.to_csv("web_traffic_anomalies.csv", index=False)