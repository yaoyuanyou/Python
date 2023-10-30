import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('mango.csv')

plt.title("Yearly Mango Production: Australia vs Top 3")
plt.ylabel("Annual Production Capacity (Tonnes)")
