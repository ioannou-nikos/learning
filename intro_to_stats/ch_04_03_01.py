# Import standard packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def univariate_plots(plot_type, x=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if plot_type == 'scatter':
        ax.plot(x, '.')
        plt.title('Scatter Plot')
        plt.xlabel('Datapoints')
        plt.ylabel('Values')
        plt.show()
    elif plot_type == 'histogram':
        ax.hist(x, bins=25)
        plt.title('Histogram (25 bins)')
        plt.xlabel('Data Values')
        plt.ylabel('Frequency')
        plt.show()
    elif plot_type == 'kde':
        sns.kdeplot(x)
        plt.title('Kernel Density Estimate')
        plt.xlabel('x')
        plt.ylabel('Frequency Function')
        plt.show()
    elif plot_type == 'cumfreq':
        ax.plot(stats.cumfreq(x, numbins=100)[0])
        plt.title('Cumulative Frequency')
        plt.xlabel('Data Values')
        plt.ylabel('CumFreq')
        plt.show()
    elif plot_type == 'errorbar':
        index = np.arange(5)
        y = index**2
        errorBar = index/2  # just for demonstration
        ax.errorbar(index, y, yerr=errorBar, fmt='o', capsize=5, capthick=3)
        plt.title('ErrorBar plot for simple data')
        plt.xlabel('Values')
        plt.ylabel('Bars')
        plt.show()
    elif plot_type == 'boxplot':
        ax.boxplot(x, sym='*')
        plt.title('Box Plot')
        plt.xlabel('Values')
        plt.show()
    elif plot_type == 'violin':
        # Generate the data
        nd1 = stats.norm
        data1 = nd1.rvs(size=(100))
        nd2 = stats.norm(loc=3, scale=1.5)
        data2 = nd2.rvs(size=(100))

        # Use pandas and the seaborn package for violin plot
        df = pd.DataFrame({'Girls':data1, 'Boys':data2})
        sns.violinplot(df)
        plt.show()
    elif plot_type == 'groupedbarcharts':
        df = pd.DataFrame(np.random.rand(10, 4),
                          columns=['a', 'b', 'c', 'd'])
        df.plot(kind='bar', grid=False)
        plt.show()
    elif plot_type == 'pie':
        txt_labels = 'Cats', 'Dogs', 'Frogs', 'Others'
        fractions = [45, 30, 15, 10]
        offsets = (0, 0.05, 0, 0)
        ax.pie(fractions, explode=offsets, labels=txt_labels,
               autopct='%1.1f%%', shadow=True, startangle=90,
               colors=sns.color_palette('muted'))
        plt.axis('equal')
        plt.show()



# Generate the data
data = np.random.randn(500)

# Plot command start ----------------------

# Scatter plot
univariate_plots('scatter', x=data)

# Histogram
univariate_plots('histogram', x=data)

# Kernel Density Estimate (KDE) plot
univariate_plots('kde', x=data)

# Cumulative Frequencies Plot
univariate_plots('cumfreq', x=data)

# Error Bar plot
univariate_plots('errorbar', x=data)

# Box Plots
univariate_plots('boxplot', x=data)

# Violin Plot
univariate_plots('violin')

# Grouped Bar Charts
univariate_plots('groupedbarcharts')

# Pie Charts
univariate_plots('pie')
# Plot command end ------------------------

