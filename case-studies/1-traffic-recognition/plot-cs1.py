import matplotlib.pyplot as plt
import numpy as np
import math, ast

x = np.arange(43)

y=[]
for i in range(1,11):
	fl = 'res-conv1/b'+str(i)+'0/res.csv'
	f = open(fl)
	l = f.readlines()
	a = l[0]
	a = ast.literal_eval(a)
	y.append(a)
	f.close()

fig, ax = plt.subplots()

ax.scatter(x, y[0], label="10")
for i in range(1,10):
	ax.scatter(x, y[i], label=(i+1)*10)

ax.legend(prop={'size': 12})
plt.xticks(x)
ax.set(xlabel="Class indices of traffic signs", ylabel="Number of correct predictions (out of 30)",
	title="Correct predictions for varying amount of bitflips vs GTSRB dataset classes")
ax.grid()

# fig.savefig("TFI2-CS1.png")
plt.show()