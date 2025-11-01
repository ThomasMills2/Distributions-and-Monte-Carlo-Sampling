#Expected PDF of the possible outcome from the sum of two throws of a 10 sided dice
import numpy as np
import matplotlib.pyplot as plt

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
