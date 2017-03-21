import pandas as pd
import numpy as np

data = pd.read_csv("datasets/raw.csv")

artist = data["artist_name"].unique()
tag = data["tag_name"].unique()
track = data["track_name"].unique()
