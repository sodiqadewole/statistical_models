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
