"""Quick analysis of the data used in traning."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


with open("../../data/scenario_nyc4x4.json", "r") as f:
    dat = f.read()


dat = json.loads(dat)
df = pd.DataFrame(dat["demand"])

mean_demand = df.groupby("time_stamp")["demand"].mean()

plt.plot(mean_demand)
plt.show()

df.groupby(["origin", "destination"]).agg({"travel_time": "unique"})

dest_mask = (df["origin"] == 0) & (df["destination"] == 0)
plt.plot(df[dest_mask].time_stamp, df[dest_mask].travel_time)


plt.figure(figsize=(10, 6))
plt.scatter(df["travel_time"], df["price"], alpha=0.2, s=2)
plt.axline((0, 0), slope=1, color="red")
plt.xlabel("Travel time")
plt.ylabel("Price")
plt.title("Demand vs. price")
plt.show()

# How much of the variance of the price is explained by time?
lr = LinearRegression()
lr.fit(df[["travel_time"]], df["price"])
lr.score(df[["travel_time"]], df["price"])
