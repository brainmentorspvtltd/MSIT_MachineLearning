import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('dataset/SaratogaHouses.csv')

x = dataset["livingArea"].values
y = dataset["landValue"].values
z = dataset["price"].values

fig = plt.figure()
fig = Axes3D(fig)
fig.plot(x,y,z,'o')
plt.xlabel("livingArea")
plt.ylabel("landValue")

plt.show()
