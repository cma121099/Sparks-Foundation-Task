#!/usr/bin/env python
# coding: utf-8

# # Task 1: Prediction using Supervised ML
# # GRIP APRIL 2022
# # Author: CHANDRA MOHAN ANAND
# ### Predict the percentage of a student based on no. of study hours. A simple linear regression task that involves two variables.

# ### Step 1: Importing the required libraries

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2:  Importing the data

# In[5]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")
df.head(10)


# In[6]:


df.describe()


# ### Step 3: Checking for the correlation between dependent and explanatory variable

# In[7]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Step 4: Defining the variables

# In[10]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# ### Step 5: Splitting the variables into training and test set

# In[11]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### Step 6: Applying linear regression model under supervised learning algorithm

# In[12]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[13]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[14]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[15]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### Step 7: Prediction of score

# In[16]:


hours=9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Step 8: Evaluation of the model

# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# # At this point TASK 1 is finished.
# # Thank you.
