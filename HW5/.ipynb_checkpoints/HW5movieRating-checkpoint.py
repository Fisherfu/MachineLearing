import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
# Step 1: Load the dataset
file_path = r'./movieRating.csv'

data = pd.read_csv(file_path)

data.head()