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

# I would suggest excluding rows 738 and 739 altogether.

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
Data exploration for model selection
"""
import matplotlib.pyplot as plt

#plt.hist(df_elem[df.columns[4]], 50)
plt.boxplot(df_elem[df.columns[7]])


#%%
"""
Model selection

Our data is structured, but we only have a limited amount of it. Random forest regressor is robust with high dimensional data and using it doesn't require scalarization or normalisation. 
"""

"""
Yield strength
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

n = len(df_ys.columns)
X = df_ys[df_ys.columns[0:n]]
y_ys = df_ys[df_ys.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y_ys, test_size=0.33)
ys_regr = RandomForestRegressor(max_depth=5, random_state=0)
ys_regr.fit(X_train, y_train)

#%%
from sklearn.metrics import r2_score

y_ys_pred = ys_regr.predict(X_test)
r2 = r2_score(y_test, y_ys_pred)

#%%
from sklearn.model_selection import cross_val_score

print(f'R2: {r2}')
print(f'K-Fold: {np.mean(cross_val_score(ys_regr, X, y_ys))}')

#%%
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [5, 10, 20, 50, 100],
              'max_depth': [None, 2, 5, 10, 20]}
ys_regr = RandomForestRegressor()
clf = GridSearchCV(
        estimator=ys_regr,
        param_grid=param_grid,
        return_train_score=True).fit(X, y_ys)

#%%
clf_results = clf.cv_results_
top_param = np.argmax(clf_results['mean_test_score'])
print(clf_results['params'][top_param])

#%%
"""
Ultimate tensile strength
"""
from sklearn.gaussian_process import GaussianProcessRegressor

X_uts1 = df_uts[df_uts.columns[0:n]]
y_uts1 = df_uts[df_uts.columns[-1]]
X_uts2 = df_uts2[df_uts2.columns[0:n]]
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

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_uts_scaled = scaler.fit_transform(X_uts1)
pca = PCA(n_components=3)
X_uts_scaled = pca.fit_transform(X_uts_scaled)

#%%
print(pca.get_feature_names_out(X_uts_scaled.columns))
#%%
clf2 = GridSearchCV(
    estimator=uts_regr, 
    param_grid=param_grid, 
    return_train_score=True).fit(X_uts_scaled, y_uts1)
clf2_res = clf2.cv_results_

"""
Based on these results Gaussian process is not suitable for this data even with PCA dimension reduction
"""


















