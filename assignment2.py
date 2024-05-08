#%%
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#%% Load the data
df_raw = pd.read_excel("C:/Users/Omistaja/Documents/TIES5830/metalapplication.xls", header=1)
df = df_raw.copy()

#%% Inspect the data
#print(df.head())
print(df.shape)
print(df.columns)
print(df[df.columns[0]].count())
print(df.dtypes)
# Count the number of non-empty cells
print(df[df.columns[0:len(df.columns)]].count())


#%% Inspect if columns Nb and Nb.1 are identical
#print(df.columns)
df_nb = df["Nb"]
df_nb1 = df["Nb.1"]
print(df_nb.equals(df_nb1))

#%%
df_nb_compare = df_nb.eq(df_nb1)
print(df_nb_compare.loc[df_nb_compare == False])
del df_nb, df_nb1, df_nb_compare

#%% Inspect if columns Ti and Ti.1 are identical
df_ti = df["Ti"]
df_ti1 = df["Ti.1"]
print(df_ti.equals(df_ti1))
del df_ti, df_ti1

#%% Inspect if columns V and V.1 are identical
print(df['V'].equals(df['V.1']))
#%%
df_v_compare = df['V'].eq(df['V.1'])
print(df_v_compare.loc[df_v_compare == False])
del df_v_compare
print(df[['V', 'V.1']].loc[107:111])
print(df[['V', 'V.1']].loc[741])

#%%
print(df[['Nb', 'Nb.1']].loc[738:739])

# Exclude rows 738 and 739 altogether.

#%%
"""
Data preprocessing
"""

# Drop empty and unclear rows and duplicate columns
df = df.drop(labels=[738, 739, 740, 741], axis='index')

# Drop duplicate columns, data number and paper number columns
columns_to_drop = list(df.columns[3:5])
columns_to_drop.append('Nb.1')
columns_to_drop.append('Ti.1')
columns_to_drop.append('V.1')
print(columns_to_drop)
df = df.drop(columns_to_drop, axis='columns')


#%%
"""
Regarding the concentrations of alloying elements, it is reasonable to assume
that a value of 0 or NaN means that that element was not used.
Empty cells in element concentration variables can therefore be filled
with zeros.

We split the elements into their own DataFrame.
"""
print(df.columns.get_loc('Sn'))
df_elem = df.iloc[:, 0:df.columns.get_loc('Sn')+1]

#%%
print(df_elem.iloc[:, 0:df.columns.get_loc('Sn')+1].isna().any())

#%%
df_elem.iloc[:, 0:df.columns.get_loc('Sn')+1].fillna(0, inplace=True)
print(df_elem.iloc[:, 0:df.columns.get_loc('Sn')+1].isna().any())

#%%
print(df_elem.dtypes)

#%%
# Change columns to float if possible. 
# Otherwise save the column with the exception.
object_columns = []
for c in df_elem.columns:
    try:
        df_elem[c].astype('float', copy=False)
    except:
        object_columns.append(c)
        
#%%
# Test what type are the cells that give an exception
print(object_columns)
print(type(df_elem['Mn'][676]))
print(type(df_elem['Mn'][677]))
print(type(df_elem['C'][676]))
print(type(df_elem['C'][677]))
#%%
# Find rows with values that are not of type int or float

object_columns = {c: [] for c in object_columns}
for c in object_columns:
    for i, val in enumerate(df_elem[c]):
        if type(val) is not int and type(val) is not float:
            object_columns[c].append(i)
print(object_columns)

# Note to self: THIS ONLY WORKED BECAUSE OF LUCK.
# ROW ORDINAL NUMBER IS NOT NECESSARILY THE SAME AS INDEX

#%%
for c in object_columns:
    print(df_elem[c][object_columns[c]])
    
#%%
"""
Ambiguous values cannot be accepted or modified without expertise in the field. Rows containing these values have to be dropped.
"""
rows = []
for c in object_columns:
    [rows.append(row) for row in object_columns[c]]

rows = np.unique(rows)
print(rows)
df_elem = df_elem.drop(rows, axis='index')

#%% Check for nonzero counts

print(df_elem.gt(0).sum(axis=0))
# print([pd.to_numeric(df[df.columns[i]], errors='coerce').gt(0).sum(axis=0) for i in range(df.columns.get_loc('Sn')+1)])

#%% Merge columns Nb and Cb as they are the same element

df_elem['Nb'] = df_elem['Nb'].mask(df_elem['Nb'] == 0, df_elem['Cb'])
df_elem.drop(['Cb'], axis='columns', inplace=True)
df_elem.drop(['Sn'], axis='columns', inplace=True)

#%% Convert values to float

df_elem = df_elem.astype('float')
print(df_elem.dtypes)

#%%
"""
Create a DataFrame for yield strength
"""
print(df[['YS(Mpa)', 'UTS(Mpa)', '%EL']].count())
df_ys = df_elem.copy()
ys = df['YS(Mpa)']
# Drop the rows that were deemed not useful
ys = ys.drop(rows)
df_ys['YS'] = ys
#%%
del ys

#%%
print(df_ys[df_ys.columns[-1]].count())
#%%
print(len(np.where(df_ys.iloc[:,-1].isna() == True)[0]))
print(df_ys[df_ys.columns[-1]].count() + len(np.where(df_ys.iloc[:,-1].isna() == True)[0]))
print(df_ys['YS(Mpa)'][86:91])

#%% Drop rows without response variable value
df_ys = df_ys.dropna(axis='index')

#%% Convert values to float
df_ys = df_ys.astype('float')
print(df_ys.dtypes)

#%%
"""
Create a DataFrame for ultimate tensile strength
"""
df_uts = df_elem.copy()
uts = df['UTS(Mpa)']
uts = uts.drop(rows)

#%%
df_uts['UTS'] = uts
del uts
print(df_uts[df_uts.columns[-1]].count())
df_uts = df_uts.astype('float')

#%%
"""
When trying to convert all values to float we notice that the UTS column also contains "bad" values: namely ranges and estimates (denoted with ~). We can either replace these with means and exact values or delete the rows. 
"""

# Create boolean mask for bad rows
uts_bad_rows = [type(val) != int and type(val) != float for val in df_uts['UTS']]

#%% Get index of bad rows

print(df_uts[uts_bad_rows])
uts_bad_rows = df_uts.index[uts_bad_rows]

#%% Convert bad values to means and exact values

import re

df_uts2 = df_uts.copy()

for row_index in uts_bad_rows:
    if '-' in df_uts['UTS'][row_index]:
        values = [float(x) for x in df_uts['UTS'][row_index].split('-')]
        df_uts['UTS'][row_index] = np.mean(values)
    else:
        number = float(re.sub("[^0-9.]", "", df_uts['UTS'][row_index]))
        df_uts['UTS'][row_index] = number



#%% Drop bad rows from second DataFrame and convert values to float

print(list(uts_bad_rows))
df_uts2.drop(list(uts_bad_rows), axis='index', inplace=True)
df_uts = df_uts.astype('float')
df_uts2 = df_uts2.astype('float')

#%% Drop rows with no response variable value

df_uts = df_uts.dropna(axis='index')
df_uts2 = df_uts2.dropna(axis='index')

#%%
"""
Create DataFrame for elongation before facture
"""

df_el = df_elem.copy()
el = df['%EL']
el = el.drop(rows)

#%% Check if response variable contains bad values

print([val for val in el if type(val) != int and type(val) != float])


#%% Create boolean mask and get index of bad rows

el_bad_rows = [type(val) != int and type(val) != float for val in el]
el_bad_rows = list(el.index[el_bad_rows])

#%% Convert bad values to exact values        

for i in el_bad_rows:
    number = float(re.sub("[^0-9.]", "", el[i]))
    el[i] = number

#%% Add %EL to DataFrame

df_el['%EL'] = el
del el

#%% Drop rows without response variable value and convert the values to float

df_el = df_el.dropna(axis='index')
df_el = df_el.astype('float')


#%%
"""
Data splitting:

Yield strength
"""
n = len(df_ys.columns)
X_ys = df_ys[df_ys.columns[0:n-1]]
y_ys = df_ys[df_ys.columns[-1]]

"""
Ultimate tensile strength
"""

X_uts1 = df_uts[df_uts.columns[0:n-1]]
y_uts1 = df_uts[df_uts.columns[-1]]
X_uts2 = df_uts2[df_uts2.columns[0:n-1]]
y_uts2 = df_uts2[df_uts2.columns[-1]]

uts_data = {'uts1': [X_uts1, y_uts1],
            'uts2': [X_uts2, y_uts2]}

"""
Elongation length
"""

X_el = df_el[df_el.columns[0:n-1]]
y_el = df_el[df_el.columns[-1]]


#%%

"""
Model selection 
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

X = [X_ys, X_uts1, X_el]
y = [y_ys, y_uts1, y_el]

models = [
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    GaussianProcessRegressor(),
    AdaBoostRegressor(),
    SVR()]

names = [
    'Random Forest',
    'Gradient Boosting',
    'Gaussian Process',
    'Ada Boost',
    'Support Vector Regression']

res_full_split = {}
for i, model in enumerate(models):
    scores = []
    for j in range(len(X)):
        
        X_train, X_test, y_train, y_test = train_test_split(X[j], y[j], test_size=0.33)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        scores.append(r2)
        #mse = mean_squared_error(y_test, model.predict(X_test))
        #scores.append([round(r2, 5), round(mse, 5)])
        
    res_full_split.update({names[i]: scores})
 
    
res_full_cross = {}
for i, model in enumerate(models):
    scores = []
    for j in range(len(X)):
        scores.append(np.mean(cross_val_score(model, X[j], y[j])))
    res_full_cross.update({names[i]: scores})


#%%
"""
Models trained with PCA
"""
X_pca = []
for j in range(len(X)):
    scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X[j])
    pca = PCA(n_components=0.5)
    #X_scaled = pca.fit_transform(X_scaled)
    #print(X_scaled.shape[1])
    #X_pca.append(X_scaled)

#%%
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
pca = PCA(n_components=0.5)

res_scaled_split = {}
for i, model in enumerate(models):
    scores = []
    for j in range(len(X)):
        X_train, X_test, y_train, y_test = train_test_split(X[j], y[j], test_size=0.33)
        pipe = Pipeline([
            ('scaler', scaler), 
            ('pca', pca), 
            ('model', model)])
        pipe.fit(X_train, y_train)
        
        r2 = r2_score(y_test, pipe.predict(X_test))
        scores.append(r2)
        #mse = mean_squared_error(y_test, model.predict(X_test))
        #scores.append([round(r2, 5), round(mse, 5)])
        
    res_scaled_split.update({names[i]: scores})
        

res_scaled_cross = {}
for i, model in enumerate(models):
    scores = []
    for j in range(len(X_pca)):
        pipe = Pipeline([
            ('scaler', scaler), 
            ('pca', pca), 
            ('model', model)])
        scores.append(np.mean(cross_val_score(pipe, X[j], y[j])))
        
    res_scaled_cross.update({names[i]: scores})

        
#%%

print('Scores with full data and train-test split:')
print(res_full_split)
print()
print('Scores with full data and cross validation:')
print(res_full_cross)
print()
print('Scores with scaling and PCA, test-train split:')
print(res_scaled_split)
print()
print('Scores with scaling and PCA, cross validation:')
print(res_scaled_cross)

#%%
"""
Save scores in a DataFrame and get best scores and models for each dataset
"""
scores = [res_full_split, res_full_cross, res_scaled_split, res_scaled_cross]
scores_df = pd.DataFrame(res_full_split).T
scores_df.columns = ['YS', 'UTS', 'EL']
scores_df = scores_df.rename_axis('Model').reset_index()
scores_df['Type'] = np.ones(scores_df.shape[0])
for i in range(1, len(scores)):
    df_temp = pd.DataFrame(scores[i]).T
    df_temp.columns = ['YS', 'UTS', 'EL']
    df_temp = df_temp.rename_axis('Model').reset_index()
    df_temp['Type'] = np.ones(df_temp.shape[0])+i
    print(df_temp)
    scores_df = pd.concat([scores_df, df_temp], ignore_index=True)

#%%
print(scores_df)

#%%

print(scores_df.loc[np.argmax(scores_df['YS'])])
print(scores_df.loc[np.argmax(scores_df['UTS'])])
print(scores_df.loc[np.argmax(scores_df['EL'])])

"""
We see that the best test scores are found on a simple test-train split. Random Forest Regressor works fairly well on all datasets.
"""
    
#%% 
"""
Feature extraction test with Random Forest
"""
important_features = {}
feature_values = []
feature_sorted_idx = []
#feature_importance = []
data_names = ['X_ys', 'X_uts', 'X_el']

for i, x in enumerate(X):
    X_train, X_test, y_train, y_test = train_test_split(x, y[i], test_size=0.33)
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    feature_sorted_idx.append(sorted_idx)
    feature_values.append(feature_importance)
    important_features.update({data_names[i]:np.array(df_ys.columns)[sorted_idx]})
    
    
#%%
"""
Visualize feature importance based on Random Forest
"""
import matplotlib.pyplot as plt
obj_names = ['YS', 'UTS', 'EL']
dfs = [df_ys, df_uts, df_el]
#print(important_features)
fig = plt.figure()
plt.subplot(1, 2, 1)
for i in range(len(dfs)):
    pos = np.arange(feature_sorted_idx[i].shape[0]) + 0.5
    plt.subplot(1,3,i+1)
    plt.barh(pos, feature_values[i][feature_sorted_idx[i]], align="center")
    plt.yticks(pos, np.array(df_ys.columns)[feature_sorted_idx[i]])
    plt.title(f"Feature Importance ({obj_names[i]})")
fig.tight_layout()
plt.show()


#%%
"""
Hyperparameter tuning


Yield strength
"""

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [5, 10, 20, 50, 100],
              'max_depth': [None, 2, 5, 10, 20]}
ys_regr = RandomForestRegressor()

# Use 5-fold cross validation for each combination of parameters
clf = GridSearchCV(
        estimator=ys_regr,
        param_grid=param_grid,
        return_train_score=True).fit(X_ys, y_ys)

#%%
clf_results = clf.cv_results_
top_param_index = np.argmax(clf_results['mean_test_score'])
top_param_ys = clf_results['params'][top_param_index]
print(f'Best parameters for yield strength: {top_param_ys}')

#%%
"""
Ultimate tensile strength
"""
from sklearn.gaussian_process import GaussianProcessRegressor

X_uts1 = df_uts[df_uts.columns[0:n-1]]
y_uts1 = df_uts[df_uts.columns[-1]]
X_uts2 = df_uts2[df_uts2.columns[0:n-1]]
y_uts2 = df_uts2[df_uts2.columns[-1]]

uts_data = {'uts1': [X_uts1, y_uts1],
            'uts2': [X_uts2, y_uts2]}

uts_regr = GaussianProcessRegressor()
param_grid = {'n_restarts_optimizer': [0,1,2,3]}

uts_results = []
for name, data in uts_data.items():
    clf2 = GridSearchCV(
        estimator=uts_regr, 
        param_grid=param_grid, 
        return_train_score=True).fit(data[0], data[1])
    uts_results.append(clf2.cv_results_)
    
# Bad results

"""
Based on these results Gaussian process is not suitable for this data even with the PCA dimension reduction.
"""

#%%
"""
UTS with Ada Boost regressor
"""
from sklearn.ensemble import AdaBoostRegressor
param_grid={'n_estimators': [10, 20, 50, 70, 100],
            'loss': ['linear', 'square', 'exponential'],
            'learning_rate': [1.0, 3.0, 5.0]}
uts_regr = AdaBoostRegressor(RandomForestRegressor())
clf3 = GridSearchCV(
    estimator=uts_regr, 
    param_grid=param_grid,
    return_train_score=True).fit(X_uts1, y_uts1)
clf3_res = clf3.cv_results_


#%%
"""
UTS with RF
"""

param_grid = {'n_estimators': [5, 10, 20, 50, 100],
              'max_depth': [None, 2, 5, 10, 20]}
uts_regr = RandomForestRegressor()

clf4 = GridSearchCV(
    estimator=uts_regr, 
    param_grid=param_grid, 
    return_train_score=True).fit(X_uts1, y_uts1)

#%%

clf4_results = clf4.cv_results_
top_param_index = np.argmax(clf4_results['mean_test_score'])
top_param_uts = clf4_results['params'][top_param_index]
print(f'Best parameters for ultimate tensile strength: {top_param_uts}')


#%%
"""
EL with Random Forest
"""

#param_grid = {'learning_rate': [0.05, 0.1, 0.15],
#              'n_estimators': [50, 100, 150, 200],
#              'max_depth': [None, 3, 5, 10],
#              'max_features': [5, 10, 15, 19]}

#el_regr = GradientBoostingRegressor()

el_regr = RandomForestRegressor()
pipe = Pipeline([
    ('scaler', scaler), 
    ('pca', pca), 
    ('model', el_regr)])

param_grid = {'model__n_estimators': [5, 10, 20, 50, 100],
              'model__max_depth': [None, 2, 5, 10, 20]}

clf5 = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    return_train_score=True).fit(X_el, y_el)

#%%

clf5_results = clf5.cv_results_
top_param_index = np.argmax(clf5_results['mean_test_score'])
top_param_el = clf5_results['params'][top_param_index]
print(f'Best parameters for elongation-%: {top_param_el}')


#%%
"""
Model optimization
"""
from scipy.optimize import minimize

model_ys = RandomForestRegressor(
    max_depth=top_param_ys['max_depth'], 
    n_estimators=top_param_ys['n_estimators'])
model_ys.fit(X_ys.values, y_ys.values)

model_uts = RandomForestRegressor(
    max_depth=top_param_uts['max_depth'], 
    n_estimators=top_param_uts['n_estimators'])
model_uts.fit(X_uts1.values, y_uts1.values)

model_el = RandomForestRegressor(
    max_depth=top_param_el['model__max_depth'], 
    n_estimators=top_param_el['model__n_estimators'])
model_uts.fit(X_el.values, y_el.values)


#model_el = GradientBoostingRegressor(
#    max_depth=top_param_el['max_depth'], 
#    n_estimators=top_param_el['n_estimators'],
#    learning_rate=top_param_el['learning_rate'],
#    max_features=top_param_el['max_features'])
model_el.fit(X_el.values, y_el.values)

#%%
import itertools

def metallurgical_problem(x):
    ys_pred = -1 * model_ys.predict([x])
    uts_pred = -1 * model_uts.predict([x])
    el_pred = -1 * model_el.predict([x])
    
    return np.concatenate([ys_pred, uts_pred, el_pred]).tolist()

def reference_point_method(f, start, z: np.ndarray):
    rho = 0.00001
    w = np.array([1.5, 1, 2.5])
    bounds = itertools.repeat((0, np.Inf), X_ys.shape[1])
    
    # Optimization problem scalarized with a reference point
    obj = lambda x : max(w*f(x)-z) + rho * sum(w * f(x) - z)
    print(obj)
    #const = ({'type': 'ineq', 'fun': lambda x:  0.96-0.96/(1.09-x**2)})
    sol = minimize(obj, start, method='SLSQP', bounds=bounds)
        #,constraints=const) #, options={'disp':True})
    return sol

#%%

x0 = np.ones(X_ys.shape[1])*0.01
#x0 = np.zeros(X_ys.shape[1])
z = np.array([-100, -100, -100])
res = reference_point_method(metallurgical_problem, x0, z)

print(f'Solution: {res.x}')
obj_values = [-1*round(o, 3) for o in metallurgical_problem(res.x)]
print(obj_values)

#%%
###############################################################
# These are for testing that the functions work as intended
w = np.array([2, 2, 2])
print([r for r in metallurgical_problem(x0)])
x0 = np.ones(X_ys.shape[1])*0.5
print(max(w*metallurgical_problem(x0)-z) + 0.00001 * sum(w * metallurgical_problem(x0) - z))

print(-model_uts.predict([np.ones(X_ys.shape[1])*0.02]))
print(-model_el.predict([np.ones(X_ys.shape[1])*0.03]))



























