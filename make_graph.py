import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd

#my_data = genfromtxt('little_data.csv', delimiter=',')
column_names = ['iteration', 'step', 'loss','q','r']
df = pd.read_csv('test3.csv', sep=',', header=None,names=column_names)
my_data = df.values
df['m'] = df['r'].rolling(window=200).mean()
rows = len(df)

plt.plot(df['iteration'], df['r'])
plt.show()
#fig.savefig('q.png')

#make script
#check out jivko's code
#gcloud
#check out memory