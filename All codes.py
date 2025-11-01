import numpy as np
import matplotlib.pyplot as plt
#All codes

#%% Question 1b

dice = np.arange(1,11)
outcomes = np.add.outer(dice,dice)

probs = np.zeros(19)
for i in range(10):
    for j in range(10):
        s = outcomes[i,j]
        probs[s-2] += 1
        
probs /= 100.0

sums = np.arange(2,21)
plt.bar(sums, probs, align='center', alpha=0.5)
plt.xticks(sums)
plt.xlabel('Sum')
plt.ylabel('Probability')
plt.title('Probability Distribution of Sum of Two 10-Sided Dice')
plt.show()

#%% Qustion 1c

rng = np.random.default_rng(210055795)

n_samples = 10000
dice = rng.integers(1, 11, size=(n_samples, 2))
sums = np.sum(dice, axis=1)

counts = np.zeros(19)
for s in sums:
    counts[s-2] += 1
    
pdf = counts / float(n_samples)

sums = np.arange(2,21)
plt.bar(sums, pdf, align='center', alpha=0.5)
plt.xticks(sums)
plt.xlabel('Sum')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation of Sum of Two 10-Sided Dice')
plt.show()

total=np.sum(dice, axis=1)
mean = (np.mean(total))
sigma = (np.std(total))
print("SD", sigma)
print("Mean", mean)

#%% Question 1d

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
