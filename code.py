
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, classification_report
from sklearn.metrics import roc_curve, auc

# In[42]:


# !pwd


# In[43]:


df = pd.read_csv("/Users/bhavi/Documents/Projects/CristianoRonaldo/Cristano_Ronaldo_Final_v1/data.csv")


# In[44]:


df = df.rename(columns={'lat/lng':'lat_lng',                        'distance_of_shot.1': 'distance_of_shot_1',                         'remaining_min.1':'remaining_min_1',                         'power_of_shot.1':'power_of_shot_1',                         'knockout_match.1' : 'knockout_match_1',                         'remaining_sec.1' : 'remaining_sec_1'})


print (df[df.match_id.isnull()].shape) # match_id is never null


# for col in df.columns:
#     print (f"'{col}'", end=',')


# In[45]:


df.describe()


# In[46]:


df.corr()


# In[47]:


# pd.scatter_matrix(df[['match_event_id','remaining_min','remaining_sec', 'remaining_min_1', 'remaining_sec_1', 'distance_of_shot_1']], alpha=0.6, figsize=(10, 10), diagonal='kde')
# plt.show()


# In[48]:


'''
    Filling Missing Values of game_season by padding on previous values
'''

print(df[df.game_season.isnull()].shape)
df['game_season'] = df['game_season'].fillna(method='pad')

df['game_season'] = df['game_season'].apply(
    lambda x: x.split('-')[0]
)

# print(df[df.game_season.isnull()].shape)


# In[49]:


'''
    Filling Missing Values of date_of_game by padding on previous values
'''

print(df[df.date_of_game.isnull()].shape)
df['date_of_game'] = df['date_of_game'].fillna(method='pad')

print(df[df.date_of_game.isnull()].shape)

'''
    Generating new features for day month and year
'''

df['year'] = df['date_of_game'].apply(
    lambda x: x.split('-')[0]
)

df['month'] = df['date_of_game'].apply(
    lambda x: x.split('-')[1]
)

df['day'] = df['date_of_game'].apply(
    lambda x: x.split('-')[2]
)


# In[50]:


'''
    Filling Missing Values of lat_lng by padding on previous values
'''

print(df[df.lat_lng.isnull()].shape)
df['lat_lng'] = df['lat_lng'].fillna(method='pad')

print(df[df.lat_lng.isnull()].shape)

'''
    lat_lng is categorical based on arena of match - we can label encode it
'''
print(df.lat_lng.unique().shape)
df['lat_lng'] = df['lat_lng'].astype('category').cat.codes


# In[51]:


# Match Event Id

# for i in range(2, len(df[df.match_event_id.isnull()])):
#     df.loc[i, 'match_event_id'] = df.loc[i-1, 'match_event_id']

'''
    Fill Missing Match Event Id By Looking at Row Above And Below with same Game Id,
    pass in case of any error.
'''
for index, row in df[df.match_event_id.isnull()].iterrows():
    try:
        if df.loc[index - 1, 'match_id'] == row['match_id']:
            df.loc[index, 'match_event_id'] = int(df.loc[index-1, 'match_event_id']) + 1
        elif df.loc[index + 1, 'match_id'] == row['match_id']:
            df.loc[index, 'match_event_id'] = int(df.loc[index+1, 'match_event_id']) + 1
        else:
            df.loc[index, 'match_event_id'] = int(df.loc[index-1, 'match_event_id']) + 1
    except:
        pass

'''
    Fill Remaining Rows(only 10) with mode value
'''
df['match_event_id'] = df['match_event_id'].fillna(method='pad')
# df['match_event_id'].fillna(df['match_event_id'].mode()[0], inplace=True)


# In[52]:


'''
    Filling Missing Valued For Knockout Matches
'''
def fillKnockoutMatches(row):
    if row.name >= 26198:
        return 1
    else:
        return 0
    
df['knockout_match'] = df['knockout_match'].fillna(method='pad')


# In[53]:


'''
    Filling some Distance Of Shot Feature values based on another feature by the same name.
    Second feature has noise in form of decimal numbers, so making sure that is not used
'''

df['distance_of_shot'] = df.apply(
    lambda row: row['distance_of_shot_1'] if np.isnan(row['distance_of_shot']) and (row['distance_of_shot_1']).is_integer() else row['distance_of_shot'],
    axis=1
)
# Filling remaing missing distance values with mean
df['distance_of_shot'].fillna(df['distance_of_shot'].mean(), inplace=True)


df['remaining_min'] = df.apply(
    lambda row: row['remaining_min_1'] if np.isnan(row['remaining_min']) and (row['remaining_min_1']).is_integer() else row['remaining_min'],
    axis=1
)
'''
    Time Remaining has no significant correlation with any of the features. Replacing missing 
    values of it by mean value of feature
'''
df['remaining_min'].fillna(df['remaining_min'].mean(), inplace=True)


# In[54]:


'''
    Preprocessing area field to contain only area abbreviation
'''
def findArea(x):
    if type(x) == float and np.isnan(x):
        return x
    else:
        return x.split('(')[1].split(')')[0]

df["area_of_shot"] = df["area_of_shot"].apply(findArea)

'''
    Label encoding area of shot. It encodes NaN value as -1. This will be handled as a different category when we One Hot Encode this while training.
'''

# print(df['area_of_shot'])
df['area_of_shot'].fillna("Unique", inplace=True) #Replacing NaN values with a unique value(quite literally)
df['area_of_shot'] = df['area_of_shot'].astype('category').cat.codes
# print(df['area_of_shot'])


# In[55]:


'''
    Preprocessing Range of Shot to give lower and upper range
'''
def processRangeOfShot(row):
    range_of_shot = str(row['range_of_shot'])
    try:
        if "Less Than" in range_of_shot:
            return 0, 8
        elif "+" in range_of_shot:
            return 24, 32
        else:
            low = range_of_shot.split('-')[0]
            high = range_of_shot.split('-')[1][0:2]
            return low, high
    except Exception as e:
        return float('NaN'), float('NaN')
    
df['lower_range'], df['upper_range'] = zip(*df.apply(processRangeOfShot, axis=1))

'''
    Filling Remaining values with mode of feature
'''

df['lower_range'].fillna(df['lower_range'].mode()[0], inplace=True)
df['upper_range'].fillna(df['upper_range'].mode()[0], inplace=True)


# In[56]:


''' 
    Processing Home/away
'''
def processHome(row):
    val = row['home/away']
    try:
        if '@' in val:
            return 0
        elif 'vs' in val:
            return 1
    except:
        return val
    
df['home'] = df.apply(processHome, axis=1)

'''
    Filling Missing Values
'''
df['home'] = df['home'].fillna(method='pad')


# In[57]:


'''
    One of 'type_of_shot' and 'type_of_combined_shot' is always present in data
'''

print(df[df.type_of_shot.isnull() & df.type_of_combined_shot.isnull()].shape)

'''
    Label encoding type_of_shot and type_of_combined_shot. It encodes NaN value as -1. This will be handled when we One Hot Encode this while training.
'''
df['type_of_shot'].fillna("Unique", inplace=True) #Replacing NaN values with a unique value(quite literally)
df['type_of_shot'] = df['type_of_shot'].astype('category').cat.codes

df['type_of_combined_shot'].fillna("Unique", inplace=True) #Replacing NaN values with a unique value(quite literally)
df['type_of_combined_shot'] = df['type_of_combined_shot'].astype('category').cat.codes


# In[58]:


'''
    Filling Missing data with mode and then label encoding
'''

df['shot_basics'].fillna(df['shot_basics'].mode()[0], inplace=True)
df['shot_basics'] = df['shot_basics'].astype('category').cat.codes


# In[59]:


for col in df.columns:
    print (f"'{col}'", end=',')


# In[60]:


# '''
#     Finding Missing Values Of power_of_shot which has high correlation with match_event_id by a logistic regression model
# '''
# def logisticRegressionForPowerOfShot(data):    
#     from sklearn.linear_model import LogisticRegression
#     linreg = LogisticRegression()

#     sub = data[data.power_of_shot.notnull()]
#     sub = sub[sub.match_event_id.notnull()]
#     X_train = sub[['match_event_id']]
#     y_train = sub[['power_of_shot']]

#     sub = data[data.power_of_shot.isnull()]
#     sub = sub[sub.match_event_id.notnull()]
#     X_test = sub[['match_event_id']]

#     from sklearn.preprocessing import StandardScaler
#     sc_X = StandardScaler()
#     X_train = sc_X.fit_transform(X_train)
#     X_test = sc_X.transform(X_test)

#     linreg.fit(X_train, y_train)

#     predicted = linreg.predict(X_test)

#     print (data[data.power_of_shot.isnull()].shape, predicted.shape)

#     print (predicted.mean(), data[data.power_of_shot.notnull()]['power_of_shot'].mean())

#     data.loc[data.power_of_shot.isnull() & data.match_event_id.notnull() , 'power_of_shot'] = predicted 
    
#     return data

# df = logisticRegressionForPowerOfShot(df.copy())

# # df = df.drop(df[df.location_x.isnull()].index)

# df['power_of_shot'].fillna(df['power_of_shot'].mode()[0], inplace=True)


# In[61]:


df[['location_x', 'location_y', 'distance_of_shot']].corr()


# In[62]:


'''
    Finding Missing Values Of Location_x by a linear regression model
'''
def linearRegressionForLocation_X(data):    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()

    sub = data[data.location_x.notnull()]
    sub = sub[sub.location_y.notnull()][sub.distance_of_shot.notnull()][sub.lower_range.notnull()][sub.upper_range.notnull()]
    X_train = sub[['location_y', 'distance_of_shot', 'lower_range', 'upper_range']]
    y_train = sub[['location_x']]

    sub = data[data.location_x.isnull()]
    sub = sub[sub.location_y.notnull()][sub.distance_of_shot.notnull()][sub.lower_range.notnull()][sub.upper_range.notnull()]
    X_test = sub[['location_y', 'distance_of_shot', 'lower_range', 'upper_range']]

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    linreg.fit(X_train, y_train)

    predicted = linreg.predict(X_test)

    print (data[data.location_x.isnull()].shape, predicted.shape)

    print (predicted.mean(), data[data.location_x.notnull()]['location_x'].mean())

    data.loc[data.location_x.isnull() &              data.location_y.notnull() &             data.distance_of_shot.notnull() &             data.lower_range.notnull() &             data.upper_range.notnull(), 'location_x'] = predicted 
    
    return data

df = linearRegressionForLocation_X(df.copy())

# df = df.drop(df[df.location_x.isnull()].index)

df['location_x'].fillna(df['location_x'].mean(), inplace=True)


# In[63]:


'''
    Finding Missing Values Of Location_y by a linear regression model
'''
def linearRegressionForLocation_Y(data):
    from sklearn.linear_model import LinearRegression

    linreg = LinearRegression()

    sub = data[data.location_y.notnull()]
    sub = sub[sub.location_x.notnull()][sub.distance_of_shot.notnull()][sub.lower_range.notnull()][sub.upper_range.notnull()]
    X_train = sub[['location_x', 'distance_of_shot', 'lower_range', 'upper_range']]
    y_train = sub[['location_y']]

    print (X_train.shape, y_train.shape)

    sub = data[data.location_y.isnull()]
    sub = sub[sub.location_x.notnull()][sub.distance_of_shot.notnull()][sub.lower_range.notnull()][sub.upper_range.notnull()]
    X_test = sub[['location_x', 'distance_of_shot', 'lower_range', 'upper_range']]

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    linreg.fit(X_train, y_train)

    predicted = linreg.predict(X_test)

    print (data[data.location_y.isnull()].shape, predicted.shape)

    print (predicted.mean(), data[data.location_y.notnull()]['location_y'].mean())

    data.loc[data.location_y.isnull() &              data.location_x.notnull() &             data.distance_of_shot.notnull() &             data.lower_range.notnull() &             data.upper_range.notnull(), 'location_y'] = predicted 
    return data

df = linearRegressionForLocation_Y(df.copy())

# df = df.drop(df[df.location_y.isnull()].index)

df['location_y'].fillna(df['location_y'].mean(), inplace=True)
    


# In[ ]:


print (df[df.match_event_id.isnull()].shape,       df[df.location_x.isnull()].shape,       df[df.location_y.isnull()].shape,       df[df.remaining_min.isnull()].shape,       df[df.power_of_shot.isnull()].shape,       df[df.knockout_match.isnull()].shape,       df[df.distance_of_shot.isnull()].shape,       df[df.area_of_shot.isnull()].shape,       df[df.shot_basics.isnull()].shape,       df[df.lower_range.isnull()].shape,       df[df.upper_range.isnull()].shape,       df[df.type_of_shot.isnull()].shape,       df[df.type_of_combined_shot.isnull()].shape)


# In[64]:


old_df = df.copy()


# In[65]:


df = old_df.copy()


# In[66]:


for col in df.columns:
    print (f"'{col}'", end=',')


# In[67]:


'''
    Univariate Selection
'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


considering_features = ['match_event_id','location_x','location_y'                        ,'remaining_min','knockout_match'                        ,'game_season', 'distance_of_shot','area_of_shot'                        ,'shot_basics','lat_lng','type_of_shot'                        ,'type_of_combined_shot', 'month','year','lower_range'                        ,'upper_range','home',                        'is_goal', 'shot_id_number'] # These 2 are not considered as training features



dt = df[df.is_goal.notnull()]

dt = dt[considering_features]

X = dt.loc[:, (dt.columns != 'is_goal') & (dt.columns != 'shot_id_number') ]
cols = X.columns

y = dt[['is_goal']]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X)
X.columns = cols


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns, dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(19,'Score'))  #print 10 best features


# In[68]:


'''
    Correlation Matrix with Heatmap
'''

import seaborn as sns

considering_features = ['match_event_id','location_x','location_y'                        ,'remaining_min','power_of_shot','knockout_match'                        ,'game_season', 'distance_of_shot','area_of_shot'                        ,'shot_basics','lat_lng','type_of_shot'                        ,'type_of_combined_shot', 'month','year','lower_range'                        ,'upper_range','home',                        'is_goal', 'shot_id_number'] # These 2 are not considered as training features


dt = df.copy()

dt = dt[considering_features]

#get correlations of each features in dataset
corrmat = dt.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g = sns.heatmap(dt[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[69]:


# considering_features = ['match_event_id','location_x','location_y'\
#                         ,'remaining_min','power_of_shot','knockout_match'\
#                         ,'game_season', 'distance_of_shot','area_of_shot'\
#                         ,'shot_basics','lat_lng','type_of_shot'\
#                         ,'type_of_combined_shot', 'month','year','lower_range'\
#                         ,'upper_range','home',\
#                         'is_goal', 'shot_id_number'] # These 2 are not considered as training features

# categorical_features = ['power_of_shot', 'knockout_match', 'game_season', 'area_of_shot' ,'shot_basics'\
#                        ,'lat_lng', 'type_of_shot', 'type_of_combined_shot', 'month','year','lower_range'\
#                         ,'upper_range','home']


considering_features = ['match_event_id','location_x','location_y'                        ,'remaining_min','knockout_match'                        , 'distance_of_shot','area_of_shot'                        ,'shot_basics','type_of_shot'                        ,'type_of_combined_shot','lower_range'                        ,'upper_range','home','month',                        'is_goal', 'shot_id_number'] # These 2 are not considered as training features

categorical_features = ['area_of_shot'                       , 'type_of_shot', 'type_of_combined_shot','lower_range'                        ,'upper_range','home','month','knockout_match']


# In[70]:


'''
    One Hot Encoding
'''
df = df[considering_features]

encoded = pd.get_dummies(data=df, columns = categorical_features)
encoded.columns = encoded.columns.str.replace(".", "_")
encoded.columns = encoded.columns.str.replace("-", "_")

for col in encoded.columns:
    print (f"'{col}'", end=',')
df = encoded
df.shape


# In[71]:


df.shape


# In[72]:


submission = df[df.is_goal.isnull()]
submission = submission.copy()
submission.loc[:, 'shot_id_number'] = submission.index + 1

submission.to_csv("TestData.csv", index=False)
data = df[df.is_goal.notnull()]
df.shape, data.shape, submission.shape


# In[73]:


def plotPrecisionRecallCurve(y_test, y_pred, average_precision):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from inspect import signature

    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))


# In[74]:


def plotROCCurve(y_test, y_pred, model):
    import sklearn.metrics as metrics
    # calculate the fpr and tpr for all thresholds of the classification
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[75]:


def evaluatePredictions(y_test, y_pred, y_train, y_pred_train, classifier):
    # Making the Confusion Matrix
    cm_test = confusion_matrix(y_test, y_pred)
    cm_train = confusion_matrix(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    average_precision = average_precision_score(y_test, y_pred)
    print (f"{cm_test} \n {accuracy_test} \n\n {cm_train} \n {accuracy_train} \n\nPrecision Recall Score = {average_precision} ")
    print (f"\n\nClassification Report\n {classification_report(y_test, y_pred)}\n")
#     plotROCCurve(y_test, y_pred, classifier)
#     plotPrecisionRecallCurve(y_test, y_pred, average_precision)

    


# In[76]:


# cols = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].columns
# X = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].values
# y = data[['is_goal']].values

# ratio = (len(y) - y.sum()) / (y.sum())
# print(ratio)


# In[77]:


cols = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].columns
X = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].values
y = data[['is_goal']].values

ratio = (len(y) - y.sum()) / (y.sum())

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [5, 6, 7, 8, 10, 11])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                         # Leave the rest of the columns untouched
# )

# X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn.utils import shuffle
X, y = shuffle(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0) # from 10 observations- 2 in test set and 8 in training set, random state is not necessary

''' Feature Scaling '''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)
# debate on scaling dummy variablesm

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0)
logistic.fit(X_train, y_train)
y_pred_train = logistic.predict(X_train)

# Predicting the Test set results
y_pred = logistic.predict(X_test)

evaluatePredictions(y_test, y_pred, y_train, y_pred_train, logistic)

from xgboost import XGBClassifier
xboost_classifier = XGBClassifier(learning_rate =0.1,
                                 n_estimators=149,
                                 max_depth=4,
                                 min_child_weight=5,
                                 gamma=0.3,
                                 reg_alpha=0.1,
                                 objective= 'binary:logistic',
                                 scale_pos_weight=1,
                                 seed=27)

xboost_classifier.fit(X_train, y_train)

y_pred_train = xboost_classifier.predict(X_train)
y_pred = xboost_classifier.predict(X_test)

evaluatePredictions(y_test, y_pred, y_train, y_pred_train, xboost_classifier)
probs = xboost_classifier.predict_proba(X_test)

# print(xboost_classifier.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
plt.figure(figsize=(20,15))
feat_importances = pd.Series(xboost_classifier.feature_importances_, index=cols)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()


# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)

# y_pred_train = classifier.predict(X_train)
# y_pred = classifier.predict(X_test)

# evaluatePredictions(y_test, y_pred, y_train, y_pred_train, classifier)


# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

# y_pred_train = classifier.predict(X_train)
# y_pred = classifier.predict(X_test)

# evaluatePredictions(y_test, y_pred, y_train, y_pred_train, classifier)


# Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# y_pred_train = classifier.predict(X_train)
# y_pred = classifier.predict(X_test)

# evaluatePredictions(y_test, y_pred, y_train, y_pred_train, classifier)


# In[78]:


X = submission.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].values
y = submission[['is_goal']].values

# ''' Feature Scaling '''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =sc_X.fit_transform(X)

probs = xboost_classifier.predict_proba(X)
solution = pd.DataFrame(columns=['shot_id_number', 'is_goal'])

solution['shot_id_number'] = submission['shot_id_number']
solution['is_goal'] = probs

solution.to_csv("bhavi_chawla_280998_code_6.csv",index=False)



#############################################################
# PARAMETER TUNING BEGINS HERE --- UNCOMMENT EVERYTHING AHEAD TO SEE
#############################################################

# # <h1>Parameter Tuning XGBoost </h1>

# # In[ ]:


# '''
#     Parameter Tuning XGBoost
# '''
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from sklearn.model_selection import cross_validate
# from sklearn import metrics   #Additional scklearn functions
# from sklearn.model_selection import learning_curve, GridSearchCV  #Perforing grid search
# # from sklearn import cross_validation, metrics   #Additional scklearn functions
# # from sklearn.grid_search import GridSearchCV   #Perforing grid search

# import matplotlib.pylab as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 12, 4

# cols = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].columns
# train = data.loc[:, (data.columns != 'shot_id_number') ]
# y = data[['is_goal']].values

# target = 'is_goal'
# IDcol = 'ID'

# def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],                          nfold=cv_folds,metrics='auc', early_stopping_rounds=early_stopping_rounds,                          verbose_eval=True)
#         alg.set_params(n_estimators=cvresult.shape[0])
    
#     #Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain['is_goal'],eval_metric='auc')
        
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
#     #Print model report:
#     print ("\nModel Report")
#     print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['is_goal'].values, dtrain_predictions))
#     print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['is_goal'], dtrain_predprob))
    
    
#     feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
    
# #Choose all predictors except target & IDcols
# predictors = data.loc[:, (data.columns != 'is_goal') & (data.columns != 'shot_id_number') ].columns

# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, train, predictors)


# # In[ ]:


# '''
# learning_rate = 0.1
# Estimators = 149,
# Accuracy : 0.6602
# AUC Score (Train): 0.727088


# learning_rate = 0.5
# Estimators = 63,
# Accuracy : 0.6528
# AUC Score (Train): 0.695635

# learning_rate = 0.3
# Estimators = 77
# Accuracy : 0.6568
# AUC Score (Train): 0.710403

# '''


# # In[ ]:


# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
#  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(train[predictors],train[target])


# # In[47]:


# gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


# # In[48]:


# param_test2 = {
#  'max_depth':[4,5,6],
#  'min_child_weight':[4,5,6]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch2.fit(train[predictors],train[target])
# gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


# # In[49]:


# param_test2b = {
#  'min_child_weight':[6,8,10,12]
# }
# gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch2b.fit(train[predictors],train[target])


# # In[51]:


# modelfit(gsearch2b.best_estimator_, train, predictors)
# gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_


# # In[52]:


# '''
#     TUNING GAMA
# '''
# param_test3 = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
#  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch3.fit(train[predictors],train[target])
# gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


# # In[53]:


# '''
#     Recalibrating number of boosting rounds
# '''
# xgb2 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=4,
#  min_child_weight=6,
#  gamma=0.3,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb2, train, predictors)


# # In[54]:


# '''
#     Tune subsample and colsample_bytree
# '''
# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
#  min_child_weight=5, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch4.fit(train[predictors],train[target])
# gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


# # In[55]:


# '''
# Here, we found 0.8 as the optimum value for 
# both subsample and colsample_bytree. Now we should try values in 0.05 interval around these.
# '''
# param_test5 = {
#  'subsample':[i/100.0 for i in range(80,95,5)],
#  'colsample_bytree':[i/100.0 for i in range(55,65,5)]
# }
# gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=235, max_depth=4,
#  min_child_weight=5, gamma=0.3, subsample=0.9, colsample_bytree=0.6,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch5.fit(train[predictors],train[target])


# # In[56]:


# gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_


# # In[ ]:


# '''
# Tuning Regularization Parameters
# '''
# param_test6 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }
# gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=235, max_depth=4,
#  min_child_weight=5, gamma=0.3, subsample=0.9, colsample_bytree=0.55,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch6.fit(train[predictors],train[target])


# # In[58]:


# gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_


# # In[59]:


# param_test7 = {
#  'reg_alpha':[0, 0.05, 0.1, 0.15, 0.2 ]
# }
# gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=255, max_depth=4,
#  min_child_weight=5, gamma=0.3, subsample=0.9, colsample_bytree=0.55,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch7.fit(train[predictors],train[target])
# gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_


# # In[60]:


# xgb4 = XGBClassifier(
#  learning_rate =0.01,
#  n_estimators=5000,
#  max_depth=4,
#  min_child_weight=5,
#  gamma=0.3,
#  subsample=0.9,
#  colsample_bytree=0.55,
#  reg_alpha=0.1,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb4, train, predictors)


# In[ ]:


# In[ ]:



################ Parameter tuning Ends here ##############



