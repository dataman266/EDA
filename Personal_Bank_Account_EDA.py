#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd

data = pd.read_csv('C:/Users/himan/Downloads/cibc (1).csv')
data


# In[33]:


data = pd.read_csv('C:/Users/himan/Downloads/cibc (1).csv', skiprows=2)
data


# In[34]:


data = data.iloc[:,0:4]
data


# In[39]:


data.dtypes


# In[37]:


# Convert 'Date' column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
data


# In[44]:


# Convert 'Withdraw' and 'Deposit' columns to numeric format
data['WITHDRAW'] = pd.to_numeric(data['WITHDRAW'], errors='coerce')
data['DEPOSIT'] = pd.to_numeric(data['DEPOSIT'], errors='coerce')


# In[45]:


# Display the first few rows
data.dtypes


# In[49]:


# Display basic stats
data.describe(include='all', datetime_is_numeric=True)


# The dataset contains 1,722 transactions.
# The date of the transactions ranges from 2021-06-22 to 2023-07-14.
# The 'Transaction Name' field contains the details about the transaction. The most frequent transaction is "SHELL C08527 MISSISSAUGA, ON" with 245 occurrences.
# The 'Withdraw' and 'Deposit' fields represent the amount of money withdrawn or deposited during a transaction. Withdrawals range from $0.01 to $8,258.98 with an average of $71.11. Deposits range from $0.01 to $10,000 with an average of $313.72.

# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(15, 5))

# Create a line plot of the number of transactions over time
data['DATE'].value_counts().sort_index().plot(kind='line')

# Set the title and labels
plt.title('No. of Transactions Over Time')
plt.xlabel('DATE')
plt.ylabel('No. of Transactions')

plt.show()


# In[53]:


# Creating a new column 'Transaction Amount' where withdrawals are considered as positive (increasing debt)
# and deposits as negative (decreasing debt)
data['Transaction Amount'] = data['WITHDRAW'].fillna(0) - data['DEPOSIT'].fillna(0)
data


# In[56]:


# Calculate average transaction amount per day
average_transaction = data.groupby('DATE')['Transaction Amount'].mean()
average_transaction


# In[63]:


# Set the figure size
plt.figure(figsize=(15, 10))

# Create a line plot of the average transaction amount over time
average_transaction.plot(kind='line')

# Set the title and labels
plt.title('Average Transaction Amount Over Time')
plt.xlabel('Date')
plt.ylabel('Average Transaction Amount')

plt.show()


# It seems there's a substantial variation in the average transaction amount over time, with some noticeable spikes and troughs. This might be due to some large transactions (either withdrawals or deposits) made on those days.

# In[67]:


# Set the figure size
plt.figure(figsize=(20, 5))

# Create a histogram of the 'Withdraw' column
plt.subplot(1, 2, 1)
sns.histplot(data['WITHDRAW'].dropna(), bins=30, kde=False)
plt.title('Distribution of Withdrawals')
plt.xlabel('Withdrawal Amount')
plt.ylabel('Frequency')


# In[74]:



# Create a histogram of the 'Deposit' column
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 2)
sns.histplot(data['DEPOSIT'].dropna(), bins=30, kde=False)
plt.title('Distribution of Deposits')
plt.xlabel('Deposit Amount')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# The histograms above show the distribution of withdrawals and deposits. Here are some observations:
# 
# Withdrawals: The majority of withdrawals are small amounts, with a few larger withdrawals.
# Deposits: The distribution of deposits is more spread out compared to withdrawals, but the majority of deposits are also in the lower range.

# In[78]:


# Top 10 transactions by volume
top_10_volume = data['TRANSACTION NAME'].value_counts().head(10)
top_10_volume


# In[79]:


# Top 10 transactions by value (withdrawals)
top_10_value_withdraw = data.groupby('TRANSACTION NAME')['WITHDRAW'].sum().nlargest(10)
top_10_value_withdraw


# In[80]:


# Top 10 transactions by value (deposits)
top_10_value_deposit = data.groupby('TRANSACTION NAME')['DEPOSIT'].sum().nlargest(10)
top_10_value_deposit


# In[82]:


# Calculate the cumulative sum of the 'Transaction Amount' column as 'Balance'
data = data.sort_values('DATE')
data['Balance'] = data['Transaction Amount'].cumsum()

# Set the figure size
plt.figure(figsize=(15, 5))

# Create a line plot of the balance over time
data.plot(x='DATE', y='Balance', kind='line')

# Set the title and labels
plt.title('Balance Over Time')
plt.xlabel('Date')
plt.ylabel('Balance')

plt.show()


# In[84]:


# Extract month and year from 'Date' and create a new column 'YearMonth'
data['YearMonth'] = data['DATE'].dt.to_period('M')


# In[86]:


# Calculate total spending per month (assuming that 'Withdraw' increases debt)
monthly_spending = data.groupby('YearMonth')['WITHDRAW'].sum()

# Set the figure size
plt.figure(figsize=(15, 5))

# Create a line plot of monthly spending
monthly_spending.plot(kind='line')

# Set the title and labels
plt.title('Monthly Spending Over Time')
plt.xlabel('Year and Month')
plt.ylabel('Total Spending')

plt.show()


# In[88]:


# Calculate the count of each transaction type
transaction_counts = data['TRANSACTION NAME'].value_counts()

# Set the figure size
plt.figure(figsize=(15, 5))

# Create a bar plot of the top 10 most common transaction types
transaction_counts.head(10).plot(kind='bar')

# Set the title and labels
plt.title('Top 10 Most Common Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')

plt.show()


# It appears that transactions at "SHELL C08527 MISSISSAUGA, ON" and "TIM HORTONS #5063 MISSISSAUGA, ON" are the most frequent.

# In[89]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset
import warnings

warnings.filterwarnings("ignore")


# In[91]:


# Resample the 'Transaction Amount' to monthly frequency
monthly_data = data.resample('M', on='DATE')['WITHDRAW'].sum()


# In[92]:


# Decompose the time series to observe trend and seasonality
decomposition = seasonal_decompose(monthly_data)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[93]:



# Plot the original data, the trend, the seasonality, and the residuals 
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(monthly_data, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[96]:



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
import itertools
import numpy as np
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Find the best parameters for the model
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        
        try:
            temp_model = SARIMAX(monthly_data,
                                 order = param,
                                 seasonal_order = param_seasonal,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            results = temp_model.fit()

            # print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            #print("Unexpected error:", sys.exc_info()[0])
            continue

# Fit the best model
best_model = SARIMAX(monthly_data,
                     order=best_pdq,
                     seasonal_order=best_seasonal_pdq,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
best_results = best_model.fit()

# Make forecast for the next 12 months
forecast = best_results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# Plot the historical data and forecast
ax = monthly_data.plot(label='Observed', figsize=(14, 7))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Spending')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




