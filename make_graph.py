import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd

#my_data = genfromtxt('little_data.csv', delimiter=',')
column_names = ['iteration', 'step', 'perc','loss','q']
df = pd.read_csv('test1.csv', sep=',', header=None,names=column_names)
my_data = df.values

rows = len(df)
print (rows)
its = []
for i in range(0,rows):
	its.append(i*100)
fig = plt.figure()

plt.scatter(df['step'], df['q'])
plt.show()
fig.savefig('q.png')

#make script
#check out jivko's code
#gcloud
#check out memory