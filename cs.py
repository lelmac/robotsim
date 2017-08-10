import csv
from matplotlib import pyplot as plt
with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
data.pop(0)
epsiode_rew = []
for i in xrange(len(data)):
    epsiode_rew.append(data[i][1]) 

print(epsiode_rew)
plt.plot(range(len(data)),epsiode_rew)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()
