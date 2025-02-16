import numpy as np
import scipy.stats as stats

# set the random seed
np.random.seed(123456)

# set sample size
n = 100

# initialize ybar to an array of length r=10000 to later store results:
r = 100000
ybar = np.empty(r)

# repeat r times
for j in range(r):
    # draw a sample and store the sample mean in pos. j=0,1,... of ybar
    # Normal distribution
    sample = stats.norm.rvs(10,2,size=n)
    # Chi2 Distribution
    #sample = stats.chi2.rvs(1,size=n)
    ybar[j] = np.mean(sample)

# print the ybar mean
print(f'The ybar mean is: {np.mean(ybar)}\n')
# compute and print the variance of ybar
print(f'The variance of ybar is: {np.var(ybar, ddof=1)}')