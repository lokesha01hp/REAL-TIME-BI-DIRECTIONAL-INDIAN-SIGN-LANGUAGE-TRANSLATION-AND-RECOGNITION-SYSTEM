import pandas as pd

csv_path = "data/gestures.csv"

df = pd.read_csv(csv_path, header=None)

num_features = df.shape[1] - 1   # all columns except the first (label)
header = ["label"] + [f"f{i}" for i in range(num_features)]

df.columns = header
df.to_csv(csv_path, index=False)

print("CSV header fixed successfully!")
