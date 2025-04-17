#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tonyyao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Read the data
conversion_df = pd.read_csv("/Users/tonyyao/Downloads/Conversion Rate/conversion_project.csv")
conversion_df.head(10)
conversion_df.drop(conversion_df[conversion_df['age'] == 111].index, inplace = True)
conversion_df.drop(conversion_df[conversion_df['age'] == 123].index, inplace = True)

# User Characteristics
country = conversion_df['country'].value_counts()
sns.barplot(x = country.index, y = country)
age = conversion_df['age'].value_counts()
sns.histplot(x = conversion_df['age'])

#After checking the age, there are outliers, delete the rows with age 111 and 123.
new_user = conversion_df['new_user'].value_counts()
new_user.index = ['New User', 'Old User']
sns.barplot(x = new_user.index, y = new_user)
source = conversion_df['source'].value_counts()
sns.barplot(x = source.index, y = source)

#Since the majority of users is new, we want to know which source is more efficient
#Also, those retention users, does the ads get them back?
#So the metircs should be users/sources segments
segment = conversion_df.groupby(['new_user', 'source'])['age'].count().reset_index()
segment.rename(columns = {'age':'count'},inplace = True)
sns.barplot(data = segment, x = 'new_user' , y = 'count', hue = 'source')
segment['percentage'] = pd.concat([segment['count'][0:3] / segment.groupby(['new_user'])['count'].sum()[0],segment['count'][3:6] / segment.groupby(['new_user'])['count'].sum()[1]])

#All page visited
all_page_visited = conversion_df['total_pages_visited']
sns.histplot(all_page_visited)


# Which source can lead to more page visited?
fig, axs = plt.subplots(3,1)
axs[0].hist(conversion_df[conversion_df['source']=='Seo']['total_pages_visited'])
axs[1].hist(conversion_df[conversion_df['source']=='Ads']['total_pages_visited'])
axs[2].hist(conversion_df[conversion_df['source']=='Direct']['total_pages_visited'])
plt.draw()

all_page_visited_groupby = conversion_df.groupby('source')['total_pages_visited'].sum()
all_page_visited_groupby_avg = all_page_visited_groupby / [88739, 72420, 155039]


# Which source can lead to more conversion?
conversion_groupby_source = conversion_df.groupby('source')['converted'].sum()
conversion_groupby_source_avg = conversion_groupby_source / [88739, 72420, 155039]

# Which age can lead to more conversion?
conversion_groupby_age = conversion_df.groupby('age')['converted'].sum()
conversion_groupby_age_avg = conversion_groupby_age / age.sort_index()

# Old or new users can lead to more conversion?
conversion_groupby_new = conversion_df.groupby('new_user')['converted'].sum()
conversion_groupby_new_avg = conversion_groupby_new / [99454, 216744]

# Which country can lead to more conversion?
conversion_groupby_country = conversion_df.groupby('country')['converted'].sum()
conversion_groupby_country_avg = conversion_groupby_country / country.sort_index()

# In what degree, does the page visited convert?
page_convert_groupby = conversion_df[conversion_df['converted'] == 1].groupby('total_pages_visited')['converted'].count() / conversion_df.groupby('total_pages_visited')['converted'].count()

conversion_df_index = conversion_df.replace({'China': 0, 'Germany': 1, 'UK': 2, 'US': 3, 'Ads': 0, 'Direct': 1, 'Seo': 2})
conversion_df_index = conversion_df_index.drop('total_pages_visited', axis = 1)


# Logistic Regression
X = conversion_df_index.iloc[:,:-1]
y = conversion_df_index.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted']))
#Not good, perhaps because of overfitting, not linearity based on recall.

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf, target_names=['Not Converted', 'Converted']))
