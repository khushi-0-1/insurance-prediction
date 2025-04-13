#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:





# Data Collection & Analysis

# In[3]:


# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('insurance.csv')


# In[4]:


# first 5 rows of the dataframe
insurance_dataset.head()


# In[5]:


# number of rows and columns
insurance_dataset.shape


# In[6]:


# getting some informations about the dataset
insurance_dataset.info()


# Categorical Features:
# - Sex
# - Smoker
# - Region

# In[7]:


# checking for missing values
insurance_dataset.isnull().sum()


# Data Analysis

# In[8]:


# statistical Measures of the dataset
insurance_dataset.describe()


# In[35]:


# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[37]:


# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[38]:


insurance_dataset['sex'].value_counts()


# In[36]:


# bmi distribution
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# Normal BMI Range --> 18.5 to 24.9

# In[39]:


# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()


# In[40]:


insurance_dataset['children'].value_counts()


# In[41]:


# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()


# In[42]:


insurance_dataset['smoker'].value_counts()


# In[43]:


# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()


# In[44]:


insurance_dataset['region'].value_counts()


# In[45]:


# distribution of charges value
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# Data Pre-Processing

# Encoding the categorical features

# In[46]:


# Encoding 'sex' column
insurance_dataset = insurance_dataset.replace({'sex': {'male': 0, 'female': 1}})
# Encoding 'smoker' column
insurance_dataset = insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}})
# Encoding 'region' column
insurance_dataset = insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}})



# Splitting the Features and Target

# In[47]:


X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[48]:


print(X)


# In[49]:


print(Y)


# Splitting the data into Training data & Testing Data

# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[51]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Linear Regression

# In[52]:


# loading the Linear Regression model
regressor = LinearRegression()


# In[53]:


regressor.fit(X_train, Y_train)


# Model Evaluation

# In[54]:


# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[55]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[56]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[57]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# Building a Predictive System

# In[34]:


age = int(input("Enter age: "))
sex = int(input("Enter sex (male = 0, female = 1): "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = int(input("Are you a smoker? (yes = 0, no = 1): "))
region = int(input("Enter region (southeast=0, southwest=1, northeast=2, northwest=3): "))

# Prepare input data
input_data = (age, sex, bmi, children, smoker, region)
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
input_data_as_dataframe = pd.DataFrame([input_data], columns=feature_names)

# Make the prediction
prediction = regressor.predict(input_data_as_dataframe)

# Print the result
print('The predicted insurance cost is USD', round(prediction[0], 2))

