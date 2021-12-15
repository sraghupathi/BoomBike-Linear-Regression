#!/usr/bin/env python
# coding: utf-8

# Objective:
# Build multi regression model for the prediction of demand for shared bikes.
# 
# Business Model:
# 
# A bike-sharing compnay known as BoomBikes offers a service where bikes are made available for thier registered or unregistered customers on a short term basis. The service would be offered as free or for a price.
# 
# Business Process:
# 
# The bikes are borrowed by customers from BoomBikes "Dock" centers which are being controlled by a technology application wherein customes would have to enter the payment informtion in order to unlock the bike and use it. The bike will be returned at any of the BookBikes "Dock" centers once ride is completed.
# 
# Problem Statement:
# 
# BoomBikes has recenlty suffered considerable dips in their revenues due to the ongoing pandemic. The compnay is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with mindful business plan to be able accelarate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.
# 
# In order to stand out from the other service providers and gain market share, BoomBike decided to understand the significant factors on which the demand for these shared bikes depends and perform predictive analysis on shared bikes demand espicially in the American market.
# 
# DataSet:
# 
# Based on various meteorological surveys and people's styles, the BookBike firm has gathered a large dataset on daily bike demands across the American market based on some factors.
# 

# In[17]:


#importing Libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Model build
import statsmodels 
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split # For Split
from sklearn.preprocessing import MinMaxScaler # For Scaling
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Supress warnings
import warnings
warnings.filterwarnings('ignore')


# # Reading and understanding the data

# In[18]:


# Reading and Understanding the data
# importing csv file

bike_day = pd.read_csv(r'C:\Users\Admin\Desktop\Upgrad\Bike Assignment_Regression\day.csv')

# Chekcing first five records
bike_day.head()


# In[19]:


# Checking for total rows and columns in bike dataframe
bike_day.shape


# In[20]:


# Checking for the null values and relevant data types
bike_day.info()


# No missing values in the data set

# In[21]:


# Changing the "season" and weathersit variable values using map function, 
# this would change from numeric variable to categorical variable

bike_day['season'] = bike_day.season.map({1:'spring',2:'summer',
                                          3:'fall',4:'winter'})

bike_day['weathersit'] = bike_day.weathersit.map({1:'clear|pclouds',
                                                  2:'light rain',
                                                  3:'lightsnow',
                                                  4:'heavyrain|snow+fog'})

# Month and weekday variables also can be converted to categorial varibles, 
# will convert based on need after reviewing the initial data visulization


# In[22]:


# Numeric values summary
bike_day.describe()


# In[23]:


# Check categorical variable values once

bike_day.head()


# # Data Visualization 

# In[24]:


# Data visualization - using pair plot to understand which variables are most correlated to 'cnt' (target variable)
# 'casual' and 'registered' variables together make 'cnt'varible, so not considering for correlation

sns.pairplot(bike_day,x_vars=['season','yr',
                              'mnth','weekday',
                              'workingday','weathersit',
                              'temp','atemp',
                              'hum','windspeed',], y_vars='cnt',size=4,aspect=1,kind='scatter')


# In[25]:


# After having initial pair plot, converting mnth, weekday to categorical variables for better visualization

bike_day['mnth'] = bike_day.mnth.map({1:'Jan',2:'Feb', 3:'Mar',4:'Apr',
                                       5:'May',6:'Jun',7:'Jul',8:'Aug',
                                       9:'Sep',10:'Oct',11:'Nov',12:'Dec'})

bike_day['weekday'] = bike_day.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',
                                            4:'Thu',5:'Fri',6:'Sat'})


# In[26]:


# Ploting again after converting month and weekday variables to categorical

sns.pairplot(bike_day,x_vars=['season','yr',
                              'mnth','weekday',
                              'workingday','weathersit',
                              'temp','atemp',
                              'hum','windspeed',], y_vars='cnt',size=4,aspect=1,kind='scatter')


# In[27]:


# Categorical varaibles analysis using box plot

plt.figure(figsize=(24,14))
plt.subplot(4,6,1)
sns.boxplot(x='workingday',y='cnt', data=bike_day)
plt.subplot(4,6,2)
sns.boxplot(x='yr',y='cnt', data=bike_day)
plt.subplot(4,6,3)
sns.boxplot(x='holiday',y='cnt', data=bike_day)
plt.subplot(4,6,4)
sns.boxplot(x='mnth',y='cnt', data=bike_day)
plt.subplot(4,6,5)
sns.boxplot(x='season',y='cnt', data=bike_day)
plt.subplot(4,6,6)
sns.boxplot(x='weathersit',y='cnt', data=bike_day)
plt.subplot(4,6,7)
sns.boxplot(x='weekday',y='cnt', data=bike_day)


# Findings:
# 
# 1. A demand increse in year 2019 for share bike service
# 2. More demand in Clear and partial cloud weather conditions
# 3. More demand in 'fall' and 'summer' seasons
# 

# In[28]:


# Understanding correlation of variables using heat map
plt.figure(figsize=(20,10))
plt.title("Correlation between variables (Categorical & Continuous)")
sns.heatmap(bike_day.corr(),cmap="YlGnBu", annot=True)
plt.show()


# Findings: Variables temp, atemp, casual, registered are postively correlated with cnt

# In[29]:


#Drop unwanted variables from the dataframe

bike_day = bike_day.drop(['instant','dteday','registered','casual','atemp'], axis=1)

# check first five records after dropping the variables

bike_day.head()


# # Data Preparation -  Dummy Varibales

# In[30]:


# Creating Dummy variables as the features "weathersit" and "season" as they have levels.

seasons_d = pd.get_dummies(bike_day['season'])


# In[31]:


#checking first five records of dummy df

seasons_d.head()


# Data Preparation - 'season' dummies
# Four colums would not required. we can drop first column as the value can be detrmined using last three colums

# In[32]:


# dropping first column from seasons_d datafram using drop_first = 'True'

seasons_dp = pd.get_dummies(bike_day['season'], drop_first = True)

# Checking column names after rename
seasons_dp.head()


# In[33]:


# Creating dummies for weathersit

weather_d = pd.get_dummies(bike_day['weathersit'])

# checking first five records
weather_d.head()


# Data Preparation - 'weathersit' dummies is having three colums. You can drop first column as the value can be detrmined using last two colums

# In[34]:


# dropping first column from weather_d datafram using drop_first = 'True'

weather_dp = pd.get_dummies(bike_day['weathersit'], drop_first = True)

# Checking column names after rename
weather_dp.head()


# In[35]:


month_d= pd.get_dummies(bike_day['mnth'])

# checking first five records
month_d.head()


# In[36]:


# dropping first column from month_d datafram using drop_first = 'True'

month_dp = pd.get_dummies(bike_day['mnth'], drop_first = True)

# Checking column names after rename
month_dp.head()


# In[37]:


week_day= pd.get_dummies(bike_day['weekday'])

# checking first five records
week_day.head()


# In[38]:


# dropping first column from month_d datafram using drop_first = 'True'

week_day = pd.get_dummies(bike_day['weekday'], drop_first = True)

# Checking column names after rename
week_day.head()


# In[39]:


# Add the result to origina bike_day dataframe

bike_day = pd.concat([seasons_dp,month_dp,weather_dp,week_day,bike_day], axis=1)

# Checking dataframe after adding dummies
bike_day.info()


# In[40]:


# Dropping the columns mnth,season, weathersit as dumy variables are created

bike_day.drop(['season','mnth','weekday','weathersit'],axis=1, inplace=True)

bike_day.info()


# In[41]:


# Checking the number of rows and columns after dropping the columns

bike_day.shape


# In[42]:


# Re-visiting the variable correlations using heatmap as few variables are dropped.
plt.figure(figsize=(20,10))
plt.title("Correlation between variables (Categorical & Continuous)")
sns.heatmap(bike_day.corr(),cmap="YlGnBu", annot=True)
plt.show()


# Findings: 
# 1. Temp, summer and yr variables are having positive correlation with 'cnt'
# 2. light rain and lightsnow varaibles are having negetive correlation with 'cnt'

# # Preparing the data for modeling (Train-test split, rescaling etc.)

# In[179]:


# Train the model
# Defining train and test set data to always have the same rows, respectively.
# import train_test_split from the sklearn.model_selection library.
# It is best practice to split the data set 70% and 30% for training and test data sets respectively.

bike_d_train, bike_d_test = train_test_split(bike_day, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[180]:


# Display train data set

bike_d_train.head()


# In[181]:


# Rescalling the features - it is required have same scale for variables in order to have better interpretation.
# There are 2 methods ( min-max scaling(0-1 and standardization (mean-0, sigma-1)))
# min-max scaling method is used for the current model

# Scler object instanciate
scaler = MinMaxScaler()


# In[182]:


# Apply scaler() to all the columns except 'Yes-No' and dummy variables
num_vars= ['temp','hum','windspeed','cnt']

bike_d_train[num_vars]= scaler.fit_transform(bike_d_train[num_vars])
bike_d_train.head()


# In[183]:


# Checking min-max values of numeric features
bike_d_train.describe()


# In[184]:


# Checking multicollinearity and correlation cofficients after scaling

plt.figure(figsize=(24, 20))
sns.heatmap(bike_d_train.corr(), cmap='YlGnBu', annot = True)
plt.show()


# #### Dividing X and Y sets for model building

# In[185]:


y_train = bike_d_train.pop('cnt')
X_train = bike_d_train


# # Model Build
# 
# Using linear regression model with Recursive Feature elimination (RFE). Linear regression function from Scikit Learn is being used for its compatibility with RFE.

# In[186]:


# importing RFE and LinearRegression 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[187]:


# Running RFE with the output number of the variables equal to 15

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm,15) # running RFE
rfe = rfe.fit(X_train, y_train)


# In[188]:


#List selected variables

list(zip(X_train.columns, rfe.support_,rfe.ranking_))


# In[189]:


#Columns RFE support is true

colm = X_train.columns[rfe.support_]
colm


# In[190]:


#Columns RFE support is false

X_train.columns[~rfe.support_]


# # Building model using statsmodel 
# 
# P-values and VIF values are calculated to understand significane for variables and how variables are correlated to each other.
# 
# P-Vale - Represents significance of variables
# VIF (Variance inflation factor) - Represents correlation between variables
# 
# These two parameters would decide which feature should be dropped
# 
# The following steps would be followed itirated before dropping each feature
# 
# 1. Running linear regression model after adding constant variable
# 2. Fit the model on training data set using statsmodel
# 3. Understand the P-values
# 4. Calculated the VIF
# 5. Use both VIF and P-value to determine which feature should be droped
# 
# Itirate the process till P-value and VIF are within acceptable range before making prediction with the model

# In[191]:


# Creating X_test dataframe with RFE selected variables
X_train_v = X_train[colm]


# In[232]:


# Adding constant variable
X_train_v = sm.add_constant(X_train_v)


# In[233]:


# Running linear regression model
lm = sm.OLS(y_train,X_train_v).fit()


# In[234]:


# Display summary of linear model

print(lm.summary())


# Weekday was intially dropped and R-squre was displayed around 0.48. After converting weekday to categorical variable and created dummy variables, R-sqaure value has been improved.

# In[235]:


# Drop the constant
X_train_v0 = X_train_v.drop(['const'], axis=1)


# In[236]:


# Calculate the VIf for new model
vif = pd.DataFrame()
X = X_train_v0
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[246]:


# Holiday has hight p-value and low VIF

X_train_v1 = X_train_v0.drop(["holiday"],axis=1)


# Re-build molde after dropping "holiday" feature

# In[247]:


#New model after dropping "holiday" feature
X_train_lm1 = sm.add_constant(X_train_v1)
lm1 = sm.OLS(y_train,X_train_lm1).fit()
print(lm1.summary())


# In[226]:


# calculate VIF for new model (after dropping 'holiday')

# Drop the constant
X_train_v1 = X_train_v1.drop(['const'], axis=1)


# In[248]:


# VIF calculatin for new model (after dropping 'holiday')

vif = pd.DataFrame()
X = X_train_v1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Humidity (hum) feature can be dropped as VIF is very high

# In[249]:


# Dropping "hum" feature as it has high VIF.

X_train_v2 = X_train_lm1.drop(['hum'], axis=1)


# In[224]:


# Buiding new model again after dropping 'hun' feature

X_train_lm2 = sm.add_constant(X_train_v2)
lm2 = sm.OLS(y_train,X_train_lm2).fit()
print(lm2.summary())


# Calculate VIF after dropping 'hum' feature

# In[130]:


# Droppoing constant

X_train_v2 = X_train_v2.drop(['const'],axis=1)


# In[131]:


# VIF calculatin for new model (after dropping 'hum' feature)

vif = pd.DataFrame()
X = X_train_v2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 'Dec' (december month) feature seems to be insignificant due to high p-value and low p-value

# In[132]:


# Dropping feature 'Dec'
X_train_v3 = X_train_lm2.drop(['Dec'], axis=1)


# In[133]:


# Building new model after dropping 'Dec' feature
X_train_lm3 = sm.add_constant(X_train_v3)
lm3 = sm.OLS(y_train,X_train_lm3).fit()
print(lm3.summary())


# In[134]:


# Calculate VIF after dropping feature 'Dec'

# Dropping constant

X_train_v3 = X_train_lm3.drop(['const'],axis=1)


# In[135]:


# Calculate VIF

vif = pd.DataFrame()
X = X_train_v3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 'Sun' feature is having high p-value and low VIF, seems to be insignificant

# In[136]:


# Dropping feature 'Sun'
X_train_v4 = X_train_lm3.drop(['Sun'], axis=1)


# In[137]:


# Building new model after dropping 'Sun' feature
X_train_lm4 = sm.add_constant(X_train_v4)
lm4 = sm.OLS(y_train,X_train_lm4).fit()
print(lm4.summary())


# In[138]:


# Calculate VIF after dropping 'Sun' feature

# Dropping constant
X_train_v4 = X_train_lm4.drop(['const'],axis=1)


# In[139]:


vif = pd.DataFrame()
X = X_train_v4
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 'Windspeed' feature is having negetive correlation on 'cnt' and low VIF, seems to be insignificant 

# In[140]:


# Building the new model after dropping feature 'windspeed'

# Dropping feature 'windspeed'.
X_train_v5 = X_train_lm4.drop(['windspeed'], axis=1)


# In[141]:


# Building new model after dropping 'windspeed' feature
X_train_lm5 = sm.add_constant(X_train_v5)
lm5 = sm.OLS(y_train,X_train_lm5).fit()
print(lm5.summary())


# In[142]:


# Calculating VIF after dropping feature 'windspeed'

# Dropping constant

X_train_v5 = X_train_lm5.drop(['const'],axis=1)


# In[143]:


# VIF calculation
vif = pd.DataFrame()
X = X_train_v5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Feature 'Nov' is having high p-value and low VIF, seems to be insignifican

# In[144]:


# Building new model by dropping feature 'Nov'

# Dropping feature 'Nov'

X_train_v6 = X_train_lm5.drop(['Nov'], axis=1)


# In[145]:


# Building new model after dropping 'Nov' feature
X_train_lm6 = sm.add_constant(X_train_v6)
lm6 = sm.OLS(y_train,X_train_lm6).fit()
print(lm6.summary())


# In[146]:


# Calculating VIF after dropping feature 'Nov'

# Dropping constant
X_train_v6 = X_train_lm6.drop(['const'],axis=1)


# In[147]:


# VIF calculation
vif = pd.DataFrame()
X = X_train_v6
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Feature 'Jul' is haveing negetive correlation and low VIF, will drop the feature and see how the model would look like.
# 

# In[148]:


# Building new model excluding 'Jul' feature

# Drop feature 'Jul'
X_train_v7 = X_train_lm6.drop(['Jul'], axis=1)


# In[149]:


# Building new model after dropping 'Jul' feature
X_train_lm7 = sm.add_constant(X_train_v7)
lm7 = sm.OLS(y_train,X_train_lm7).fit()
print(lm7.summary())


# In[150]:


# Calculating VIF after dropping feature 'Jul'

# Dropping constant
X_train_v7 = X_train_lm7.drop(['const'],axis=1)


# In[151]:


# VIF calculation
vif = pd.DataFrame()
X = X_train_v7
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Residual Analysis of train data
# 
# Will check the if the error terms are also normally distributed by plotting histogram of error terms

# In[152]:


X_train_lm7


# In[155]:


# y train predicted

y_train_pred = lm7.predict(X_train_lm7)


# ## Error terms normal distribution

# In[156]:


# Plot the histograms of error after importing matplotlib and seaborn
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
plt.figure(figsize=(10,5))
sns.distplot((y_train - y_train_pred), bins=20)
plt.show()


# Error terms are normally distributed which is one of major assumption of linear distribution

# # Prediction and Evaluation on the test set

# In[157]:


# Apply the scaling on test data
num_vars= ['temp','hum','windspeed','cnt']

# Fit on the data
bike_d_test[num_vars]= scaler.transform(bike_d_test[num_vars])

bike_d_test.head()


# ##### Dividing X_test and y_test

# In[158]:


# Dividing X_test and y_test

y_test = bike_d_test.pop('cnt')
X_test = bike_d_test
X_test.describe()


# In[160]:


# Display 
X_train_v7.columns


# In[162]:


# Creating X_test_v0 new datafram by dropping variables from X_test
X_test_v0 = X_test[X_train_v7.columns]

# Adding constand value

X_test_v1 = sm.add_constant(X_test_v0)
X_test_v1.head()


# In[163]:


# Perform predictions

y_pred = lm7.predict(X_test_v1)


# ##### Determining R-Squar and Adjusted R-Square for test 

# In[167]:


# Evaluation of R-Square for test

from sklearn.metrics import r2_score

r2score = r2_score(y_test, y_pred)
r2score


# In[170]:


# Evaluation of adjusted R-Square
# formula - r2=1-(1-R2)*(n-1)/(n-p-1)

Adjr2 = 1- (1-r2score)*(9-1)/(9-1-1)
print(Adjr2)


# # Model Evaluation

# In[172]:


# Ploting y_test and y_pred to understand the spread

fig = plt.figure()
plt.figure(figsize =(14,6))
plt.scatter(y_test,y_pred, color='green')
fig.suptitle('y_test vs y_pred', fontsize=18)
plt.xlabel('y_test', fontsize=16)
plt.ylabel('y_pred', fontsize=16)


# In[176]:


# Visualizing fit on test data 

fig = plt.figure()
plt.figure(figsize =(14,6))
sns.regplot(x=y_test, y=y_pred, ci=55, fit_reg = True, scatter_kws = {"color": 'purple'},line_kws={"color":"green"})
plt.title('y_test vs y_pred', fontsize=18)
plt.xlabel('y_test', fontsize=16)
plt.ylabel('y_pred', fontsize=14)


# #### Comparision between train and test model

#     Train Model R-Square = 0.814
#     Train model adjusted R-Square = 0.811
# 
#     Test model R-Square = 0.8063
#     Test model adjusted R-Square = 0.7786
# 
#     Difference of R-Square between train and test is 0.98%
# 
#     Difference of adjsuted R-square between train and test is 4.06% which is less than 5%
#     
# Interpretatin:
#     
# Based on the adjusted R-Square between train and test model, it seems to be good model to use to predit shared bike service demand.
# 
# Observation on Categorical varaibles
# 
# 1.	A demand increase in year 2019 for share bike service
# 2.	More demand in Clear and partial cloud weather conditions
# 3.	More demand in 'fall' and 'summer' seasons
# 4.	Saturday being demand day in a week
# 5.	Demand spike in Sep and Oct months
# 
# Temp Temperature) is the numerical variable which is having highest correlation with ‘cnt’ (target variable)
# 
# Top three features
# 1.	Temp(Temperature)
# 2.	Weathersit (winter, sprint etc)
# 3.	Weekday(Sat being most)
# 
# 
