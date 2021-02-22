data = [0.873,2.817,0.121,-0.945,-0.055,-1.436,0.360,-1.478,-1.637,-1.869]

def shapiro_wilk(data):
    from scipy.stats import shapiro
    stat, p = shapiro(data)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")

def de_agostino(data):
    from scipy.stats import normaltest
    stat, p = normaltest(data)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")

def anderson_darling(data):
    from scipy.stats import anderson
    result = anderson(data)
    print(f"stat={result.statistic:.3f}")
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f"Probably Gaussian at the {sl:.1f}% level")
        else:
            print(f"Probably not Gaussian at the {sl:.1f}% level")


if __name__ == "__main__":
    shapiro_wilk(data=data)
    de_agostino(data=data)
    anderson_darling(data=data)