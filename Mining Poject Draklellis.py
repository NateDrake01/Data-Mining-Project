# Data Mining Project - Ignatios Draklellis

# age - age (Numeric)
# female - sex (0 = male, 1 = female)
# wbho - White = 0, Hispanic = 1, black = 2, other = 3
# forborn - Foreign born (0 = foriegn born, 1 = US born)
# citizen - US citizen (0 = No-US citizen, 1 = US citizen)
# vet - Veteran (0 = No-vateren, 1 = veteran)
# married - Married (0 = Never married, 1 = married)
# marstat - Marital status (Married, Never Married, Divorced, Widowed, Separated)
# ownchild - Number of children (Numeric)
# empl - Employed (1 = employed, 0 = unemployed or not in LF)
# unem - Unemployed (0 = employed, 1 = unemployed)
# nilf - Not in labor force (0 = Not in labor force, 1 = in labor force)
# uncov - Union coverage (0 - non-Union coverage, 1 = Union coverage)
# state - state (50 states)
# educ - Education level (HS, Some college, College, Advanced, LTHS)
# centcity - Central city (0 = no Central city, 1 = Central city)
# suburb - suburbs (0 = no suburbs area, 1 = suburbs area)
# rural - rural (0 = no rural area, 1 = rural area)
# smsastat14 - Metro CBSA FIPS Code
# ind_m03 - Major Industry Recode (Educational and health services, Wholesale and retail trade, Professional and business services, Manufacturing, Leisure and hospitality, Construction, Financial activities, Other  services, Transportation and utilities, Public administration, Agriculture, forestry, fishing, and hunting, Information, Mining, and Armed Forces)
# agric - Agriculture (0 = Non-Argiculture job, 1 = Non-Agriculture job)
# manuf - Manufacturing (0 = Non-Manufacturing job, 1 = Non-Manufacturing job)
# hourslw - Hours last week, all jobs (Numeric)
# rw - Real hourly wage, 2019$ (Numeric)
# multjobn - Number of jobs (Numeric)
#
# 


#%%
# Step 1 - plot the boxplot to investigate the outlier in "age", "rw", and "houtslw" (Numuric)
import matplotlib.pyplot as plt 
import numpy as np
boxplot = df.boxplot(column=['age', 'rw', 'rural', 'forborn', 'nilf', 'hourslw', 'multjobn'])
dfGeo = df[['age', 'rw', 'rural', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]

#Recode
dfGeo.wbho[dfGeo.wbho == 'White'] = 0
dfGeo.wbho[dfGeo.wbho == 'Hispanic'] = 1
dfGeo.wbho[dfGeo.wbho == 'Black'] = 2
dfGeo.wbho[dfGeo.wbho == 'Other'] = 3

dfGeo.educ[dfGeo.educ == 'LTHS'] = 0
dfGeo.educ[dfGeo.educ == 'HS'] = 1
dfGeo.educ[dfGeo.educ == 'Some college'] = 2
dfGeo.educ[dfGeo.educ == 'College'] = 3
dfGeo.educ[dfGeo.educ == 'Advanced'] = 4


dfGeo = dfGeo.dropna()

dfGeo.tail(35)

#%%
#Step 2 - remove outlier by calculated the interquartile range (IQR). 
# IQR is calculated as the difference between the 75th and 25th percentiles or IQR = Q3 − Q1. 
# So, the numbers that are out of IQR are outlier. 
Q1 = dfGeo.quantile(0.25)
Q3 = dfGeo.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

dfGeo = dfGeo[~((dfGeo < (Q1 - 1.5 * IQR)) |(dfGeo > (Q3 + 1.5 * IQR))).any(axis=1)]
print(dfGeo.shape)

#%%
# Step 3
 merge the data (3 numuric) without outlier to original dataset by index of both the dataframes.
dfGeo = df.merge(dfGeo, left_index=True, right_index=True).dropna()

# reset index for new dataset
dfGeo = dfGeo.reset_index()

#rename age_x, rw_x, hourlw_x to age, rw, and hourlw, and drop age_y, rw_y, and hourlw_y
dfGeo = dfGeo.rename(columns={"age_x": "age", "rw_x": "rw", "hourslw_x": "hourslw"})
dfGeo = dfGeo.drop(columns=['age_y', 'rw_y', 'hourslw_y','index'])

"rural_x":"rural", "forborn_x":"forborn", "nilf_x":"nilf", "multjobn_x":"multjobn", "educ_x":"educ"
#

#Cleaned_data
dfChkBasics(dfGeo, True)
print(dfGeo.dtypes)


#%%
#Data Visuals and Graphing

#Descriptive Statistics
geoVars=dfGeo[['age','hourslw','rw']]
tab1=geoVars.describe()
print(tab1)

#relabeling
dfGeo.rural[dfGeo.rural == 0] = 'Urban'
dfGeo.rural[dfGeo.rural == 1] = 'Rural'

dfGeo.forborn[dfGeo.forborn == 0] = 'Immigrant'
dfGeo.forborn[dfGeo.forborn == 1] = 'Native'

dfGeo = dfGeo.rename(columns={"rural": "Geography", "forborn": "Foreign Born"})

#Histogram of income by Geography
geo_groups = dfGeo.groupby('Geography').groups
urb_ind = geo_groups['Urban']
rur_ind = geo_groups['Rural']
plt.hist([dfGeo.loc[urb_ind,'rw'],dfGeo.loc[rur_ind,'rw']], 20, density = 1, histtype = 'bar', label = dfGeo['Geography'].unique())
plt.xlabel('Wage ($)')
plt.ylabel('Frequency')
plt.title('Income Distributions by Geography')
plt.legend()
plt.show()


#Violin Plots
import seaborn as sns
sns.catplot(x="Geography", y="rw", hue="Foreign Born",
            kind="violin", split=False, data=dfGeo)
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Foreign and Native Born Wages Based on Geography")
plt.show()

sns.catplot(x="Geography", y="rw",
            kind="violin", split=False, data=dfGeo)
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Foreign and Native Born Wages Based on Geography")
plt.show()


# Grouped Bar Plots
sns.catplot(x="educ", y="rw", hue="Geography", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Geography")
plt.show()

sns.catplot(x="educ", y="rw", hue="Foreign Born", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Immigration")
plt.show()

sns.catplot(x="Geography", y="rw", hue="Foreign Born", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Geography and Immigration")
plt.show()


#%%
#Reload original data and encode strings to ints
dfGeo = df[['age', 'rw', 'rural', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]

#Recode
dfGeo.wbho[dfGeo.wbho == 'White'] = 0
dfGeo.wbho[dfGeo.wbho == 'Hispanic'] = 1
dfGeo.wbho[dfGeo.wbho == 'Black'] = 2
dfGeo.wbho[dfGeo.wbho == 'Other'] = 3

dfGeo.educ[dfGeo.educ == 'LTHS'] = 0
dfGeo.educ[dfGeo.educ == 'HS'] = 1
dfGeo.educ[dfGeo.educ == 'Some college'] = 2
dfGeo.educ[dfGeo.educ == 'College'] = 3
dfGeo.educ[dfGeo.educ == 'Advanced'] = 4


dfGeo = dfGeo.dropna()

dfGeo.head(25)


#%%
#Make 4:1 train/test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(dfGeo.drop('rural',axis=1), dfGeo['rural'], test_size=0.20)


#Logistic Regression
geoLogit = LogisticRegression() #instantiate Logit
geoLogit.fit(X_train,y_train)
predictions = geoLogit.predict(X_test)
#Results
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print(classification_report(y_test,predictions))
geoLogit.score(X_test, y_test)


#CV on Logit
cv_model=LogisticRegressionCV()
cv_model.fit(X_train,y_train)
cv_predictions = cv_model.predict(X_test)
#Results
print(classification_report(y_test,cv_predictions))
cv_model.score(X_test, y_test)


#%%
# Mutiple Linear Regression
from statsmodels.formula.api import ols

linearGeoModel = ols(formula='rural ~ rw + C(nilf) + hourslw + C(forborn) + C(multjobn) + C(educ) + C(wbho)', data=dfGeo).fit()

print( type(linearGeoModel) )
print( linearGeoModel.summary() )

#OLS Predictions 
modelpredicitons = pd.DataFrame( columns=['geoModel_LM'], data= linearGeoModel.predict(dfGeo)) 
print(modelpredicitons.shape)
print( modelpredicitons.head() )



# %%
#Train Test 4:1 Split
xtarget = dfGeo[['age', 'rw', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]
ytarget = dfGeo['rural'] 

print(type(xtarget))
print(type(ytarget))

#make a train-test split in 4:1 ratio. 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import glm
import statsmodels.api as sm


x_trainGeo, x_testGeo, y_trainGeo, y_testGeo = train_test_split(xtarget, ytarget, test_size = 0.2, random_state=1)

print('x_trainGeo type',type(x_trainGeo))
print('x_trainGeo shape',x_trainGeo.shape)
print('x_testGeo type',type(x_testGeo))
print('x_testGeo shape',x_testGeo.shape)
print('y_trainGeo type',type(y_trainGeo))
print('y_trainGeo shape',y_trainGeo.shape)
print('y_testGeo type',type(y_testGeo))
print('y_testGeo shape',y_testGeo.shape)


#logistic regression
geoLogitModelFit = glm(formula='rural ~ rw + C(nilf) + hourslw + C(forborn) + C(multjobn) + C(educ) + C(wbho)', data=dfGeo, family=sm.families.Binomial()).fit()
print( geoLogitModelFit.summary() )

# logistic regression model for geography with the train set, and score it with the test set.
sklearn_geoModellogit = LogisticRegression()  # instantiate
sklearn_geoModellogit.fit(x_trainGeo, y_trainGeo)
print('Logit model accuracy (with the test set):', sklearn_geoModellogit.score(x_testGeo, y_testGeo))

sklearn_geoModellogit_predictions = sklearn_geoModellogit.predict(x_testGeo)
#print(sklearn_geoModellogit_predictions)
print(sklearn_geoModellogit.predict_proba(x_trainGeo[:8]))
print(sklearn_geoModellogit.predict_proba(x_testGeo[:8]))

#results (Logit)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print('Logit model accuracy (with the test set):', sklearn_geoModellogit.score(x_testGeo, sklearn_geoModellogit_predictions))
confusion_matrix = confusion_matrix(y_testGeo, sklearn_geoModellogit_predictions)
print(confusion_matrix)
print(classification_report(y_testGeo,sklearn_geoModellogit_predictions))


#%%
# K Nearest Neighbors (KNN)
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn_split = KNeighborsClassifier(n_neighbors=k)
    knn_split.fit(x_trainGeo,y_trainGeo)
    scores = knn_split.score(x_testGeo,y_testGeo)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN_split')
plt.ylabel('Accuracy score')
plt.show()

# the comfortable KNN choice (n = 5) 
knn_split_5 = KNeighborsClassifier(n_neighbors=5) 
# instantiate with n value given
knn_split_5.fit(x_trainGeo,y_trainGeo)
# knn_split.score(x_testGeo,y_testGeo)
knn_Geopredictions = knn_split_5.predict(x_testGeo)
print(knn_Geopredictions)
print(knn_split_5.predict_proba(x_trainGeo[:8]))
print(knn_split_5.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("KNN (k value = 5)")
print()
print(f'KNN train score:  {knn_split_5.score(x_trainGeo,y_trainGeo)}')
print(f'KNN test score:  {knn_split_5.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, knn_Geopredictions))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_testGeo, knn_Geopredictions)
print(confusion_matrix)
print(classification_report(y_testGeo, knn_Geopredictions))
print() 

#%%
#Decision Trees
from sklearn.tree import DecisionTreeClassifier

dtreeGeo = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
# Fit dt to the training set
dtreeGeo.fit(x_trainGeo,y_trainGeo)
# Predict test set labels
dtreeGeoPred = dtreeGeo.predict(x_testGeo)
print(dtreeGeoPred)
print(dtreeGeo.predict_proba(x_trainGeo[:8]))
print(dtreeGeo.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("DecisionTreeClassifier: entropy(max_depth = 5)")
print()
print(f'dtree train score:  {dtreeGeo.score(x_trainGeo,y_trainGeo)}')
print(f'dtree test score:  {dtreeGeo.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, dtreeGeoPred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testGeo, dtreeGeoPred))
print(classification_report(y_testGeo, dtreeGeoPred))
print()


#%%
#SVC() for Geography model
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix

# SVC 
svcGeoAuto = SVC(gamma='auto', probability=True)
svcGeoAuto.fit(x_trainGeo,y_trainGeo)

#Predictions
svcGeoAuto_pred = svcGeoAuto.predict(x_testGeo)
print(svcGeoAuto_pred)
print(svcGeoAuto.predict_proba(x_trainGeo[:8]))
print(svcGeoAuto.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("SVC (adjust gamma: auto)")
print()
print(f'svc train score:  {svcGeoAuto.score(x_trainGeo,y_trainGeo)}')
print(f'svc test score:  {svcGeoAuto.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, svcGeoAuto_pred))
print(confusion_matrix(y_testGeo, svcGeoAuto_pred))
print(classification_report(y_testGeo, svcGeoAuto_pred))
print()


#%%
#Linear Kernel SVC
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import svm


svcGeo_linearsvc = svm.LinearSVC()
svcGeo_linearsvc.fit(x_trainGeo,y_trainGeo)
y_pred = svcGeo_linearsvc.predict(x_testGeo)


#Predictions
y_pred_linearsvc = svcGeo_linearsvc.predict(x_testGeo)
print(y_pred)
print(svcGeo_linearsvc._predict_proba_lr(x_trainGeo[:8]))
print(svcGeo_linearsvc._predict_proba_lr(x_testGeo[:8]))

# Evaluate test-set accuracy
print("SVC LinearSVC()")
print()
print(f'svc train score:  {svcGeo_linearsvc.score(x_trainGeo,y_trainGeo)}')
print(f'svc test score:  {svcGeo_linearsvc.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, y_pred_linearsvc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testGeo, y_pred_linearsvc))
print(classification_report(y_testGeo, y_pred_linearsvc))
print()



#%%
#Run Time Calculations for Geo Models
from sklearn.model_selection import cross_val_score
import timeit

%timeit -r 1 print(f'\n cv_model CV accuracy score: { cross_val_score(cv_model, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 13.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


%timeit -r 1 print(f'\n linear SVC CV accuracy score: { cross_val_score(svcGeo_linearsvc, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 22.4 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n logistic CV accuracy score: { cross_val_score(geoLogit, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 1.12 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n KNN CV accuracy score: { cross_val_score(knn_split_5, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 1.66 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n dtree CV accuracy score: { cross_val_score(dtreeGeo, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 514 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)




# %%
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

ns_probs = [0 for _ in range(len(y_testGeo))]
lr_probs = geoLogit.predict_proba(x_testGeo)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_testGeo, ns_probs)
lr_auc = roc_auc_score(y_testGeo, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_testGeo, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testGeo, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#Dtree
ns_probs = [0 for _ in range(len(y_testGeo))]
lr_probs = dtreeGeo.predict_proba(x_testGeo)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_testGeo, ns_probs)
lr_auc = roc_auc_score(y_testGeo, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_testGeo, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testGeo, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='dtreeGeo')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()





#%%
