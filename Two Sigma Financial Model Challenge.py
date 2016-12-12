import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
pd.set_option('display.max_columns', 120)

## Set our directory
os.getcwd()
os.chdir('D:\Data Updates\MyBOOKS\Removable Disk\R-Bloggers_CODES\Kaggle Competitions\Two Sigma Financial Modeling Challenge\Train')


## Get The Data
print('Shape : {}'.format(df.shape))
with pd.HDFStore('train.h5') as train:
    df = train.get('train')

## how many assets (instruments) are we tracking?
len(df.id.unique())
# how many periods? 
len(df.timestamp.unique()) 

#Let's try to create Aggregates of performance by 

market_df = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
market_df.head()

t      = market_df['timestamp']
y_mean = np.array(market_df['y']['mean'])
y_std  = np.array(market_df['y']['std'])
n      = np.array(market_df['y']['len'])

#Let's try to visualize the market return over the time period.

plt.figure()
plt.plot(t, y_mean, '.')
plt.xlabel('timestamp')
plt.ylabel('mean of y')

plt.figure()
plt.plot(t, y_std, '.')
plt.xlabel('timestamp')
plt.ylabel('std of y')

plt.figure()
plt.plot(t, n, '.')
plt.xlabel('timestamp')
plt.ylabel('portfolio size')

##take the log of the periodic mean returns and get a cumulative sum 
#for each time period to derive a fairly good approximation of a price chart for the portfolio.
simple_ret = y_mean # this is a vector of the mean of asset returns for each timestamp
cum_ret = np.log(1+simple_ret).cumsum()

portfolio_mean = np.mean(cum_ret)
portfolio_std = np.std(cum_ret)
print("portfolio mean periodic return: " + str(portfolio_mean))
print("portfolio std dev of periodic returns: " + str(portfolio_std))

#Let's take a look at some individual assets:
assets_df = df.groupby('id')['y'].agg(['mean','std',len]).reset_index()
assets_df.head()