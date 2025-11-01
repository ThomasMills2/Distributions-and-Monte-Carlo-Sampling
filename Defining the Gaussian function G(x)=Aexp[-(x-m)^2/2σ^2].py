#%% Question 1d
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed = 210055795)

arr1=rng.integers(low=1,high=11, size=(10000,10))

total=np.sum(arr1, axis=1)

plt.hist(total,bins=91)
plt.xticks([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
plt.xlabel('Possible Outcomes from Sum of 10 Dice')
plt.ylabel('Number of Outcomes')
plt.title('Monte Carlo simulation of Sum of 10000 Throws of 10-Sided Dice')
print(np.mean(total))
print(np.std(total))
x=np.arange(1,101,1)
g=500*(np.exp(-((x-55.0565)**2)/(2*(9.088009009128458**2))))
plt.plot(x,g, color='green')
plt.legend(['Gaussian Distribution','Data'])
