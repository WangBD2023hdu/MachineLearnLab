# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns

# Statistics
import statsmodels.api as sm

# Machine Learning
from sklearn.metrics import mean_squared_error, r2_score

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Loggers
import logging
logging.getLogger().setLevel(logging.ERROR)

# Get the data
# see the dataset in the lab directory
df = pd.read_csv('medium_posts.csv', usecols=['published', 'url'])
print(df.shape)
df.head()


