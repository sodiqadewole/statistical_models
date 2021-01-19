import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices

# The Guerry dataset is a collection of historical data used in support of Andre-Michel
# 1833 Essay on the Moral Statistics of France.
# The data has 86 rows and 23 columns

df = sm.datasets.get_rdataset("Guerry", "HistData").data
df.head()

# We are interested in certain variables
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df.head()
# We want to drop the missing values
df = df.dropna()
len(df)
''' We want to know whether literacy rates in the 86 French departments
are associated with per capita wagers on the Royal Lottery in th 1820s
We need to control for the level of wealth in each department, and
we also want to include a series of dummy variables on the right-hand
side of our regression equation to control for unobserved heterogeneity
due to regional effects. '''

# \beta = (X'X)^-1 X'y
# We use the patsy module to prepare design matrice using R-like formulas
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
X
# Model fit and Summary
model = sm.OLS(y, X)
res = model.fit()
print(res.summary())
dir(res)
# Diagnostics and Specification tests - Rainbow test for linearity
sm.stats.linear_rainbow(res)
# the first number is the F-statistic and the second is the p-value

# We can draw a plot of partial regression
sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],data=df, obs_labels=False)
