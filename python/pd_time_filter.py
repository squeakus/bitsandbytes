import pandas as pd
from datetime import date, timedelta

df = pd.read_csv("FullBlinds.csv")
df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"])
print(df["INSPECTION_DATE"])
today = date.today()
week_prior = today - timedelta(weeks=12)
print("week", week_prior)
df_last_week = df[df["INSPECTION_DATE"].dt.date > week_prior]


df_last_week.to_csv("last_week.csv")
