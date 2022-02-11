# %%
import pandas as pd

wafers = {"wafer": ["A", "B", "C", "D", "E", "F"], "lot": [20, 20, 19, 18, 17, 16]}
lots = {"lot": [18, 19, 20, 21], "product": ["ABC", "DEF", "GHI", "XYZ"]}

#%%


wafer_df = pd.DataFrame(wafers)
lot_df = pd.DataFrame(lots)

# %%
inner = wafer_df.merge(lot_df, on="lot", validate="many_to_one")


print(inner.head())
# %%
outer_df = full = wafer_df.merge(lot_df, on="lot", validate="many_to_one", how="outer")
unmatched_wafers = outer_df["product"].isnull().values.any()
unmatched_lots = outer_df["wafer"].isnull().values.any()
print(f"Unmatched wafers: {unmatched_wafers}")
print(f"unmatched lots: {unmatched_lots}")


# %%
