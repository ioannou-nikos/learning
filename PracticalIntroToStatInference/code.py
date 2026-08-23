# In this article, we will explore how to perform t-tests, ANOVA, 
# and Chi-Square tests in Python with practical examples and interpretations.

# The two main pillars of statistical hypothesis testing are: Estimation and
# Hypothesis Testing. 

# One-Sample T-Test
import numpy as np
from scipy import stats
# Generate random data
data = np.random.normal(100,15,30)
print(data)
# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(data, popmean=100)
print("One-Sample T-Test:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Independent Two-Sample T-Test
# Generate random data for two groups
group1 = np.random.normal(50,10,30)
group2 = np.random.normal(55,10,30)
# Perform independent two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)
print("\nIndependent Two-Sample T-Test:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Paired Sample T-Test
# Generate random data for paired samples
before = np.random.normal(70,10,30)
after = before + np.random.normal(5,5,30)
# Perform paired sample t-test
t_stat, p_value = stats.ttest_rel(before, after)
print("\nPaired Sample T-Test:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)