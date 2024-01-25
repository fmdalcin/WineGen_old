import numpy as np
import pandas as pd
import os

from preprocessing import  clean_rawdata, scaling
from model import train, predict

df = clean_rawdata()
X_scaled = scaling(df)
print(X_scaled)
