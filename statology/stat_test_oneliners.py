import statsmodels.api as sm
import pandas as pd
from scipy import stats
import numpy as np

# Load Iris dataset from statsmodels
iris = sm.datasets.get_rdataset("iris").data

# Separate species for group comparisons
setosa = iris[iris['Species'] == 'setosa']['Sepal.Length']
versicolor = iris[iris['Species'] == 'versicolor']['Sepal.Length']
virginica = iris[iris['Species'] == 'virginica']['Sepal.Length']

# One-sample t-test
print(stats.ttest_1samp(iris['Sepal.Length'], popmean=5.5))

# Independent two-sample t-test
print(stats.ttest_ind(setosa, versicolor))

# Shapiro-Wilk test for normality
print(stats.shapiro(iris['Sepal.Length']))

# One way ANOVA
anova_result = stats.f_oneway(setosa, versicolor, virginica)
print(anova_result)