import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd

#my_data = genfromtxt('little_data.csv', delimiter=',')
column_names = ['games', 'loss', 'q_mean','reward']
df = pd.read_csv('test1.csv', sep=',', header=None,names=column_names)
my_data = df.values

rows = len(df)
print (rows)
its = []
for i in range(0,rows):
	its.append(i*100)

plt.scatter(its, df['reward'])
plt.show()
plt.savefig('reward-iterations.pdf')

#make script
#check out jivko's code
#gcloud
#check out memory