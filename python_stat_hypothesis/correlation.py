data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]


def pearson(x, y):
    from scipy.stats import pearsonr
    stat, p = pearsonr(x,y)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably independent')
    else:
        print("Probably dependent")

def spearman(x, y):
    from scipy.stats import spearmanr
    stat, p = spearmanr(x,y)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably independent')
    else:
        print("Probably dependent")


def kendall(x, y):
    from scipy.stats import kendalltau
    stat, p = kendalltau(x,y)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably independent')
    else:
        print("Probably dependent")

def x_squared():
    #For categorical
    from scipy.stats import chi2_contingency
    table = [[10, 20, 30], [6,9,17]]
    stat, p, dof, expected = chi2_contingency(table)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print('Probably independent')
    else:
        print("Probably dependent")

if __name__ == "__main__":
    pearson(x=data1, y=data2)
    spearman(x=data1, y=data2)
    kendall(x=data1, y=data2)
    x_squared()