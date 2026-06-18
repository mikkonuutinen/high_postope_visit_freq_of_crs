#%%

#%%
# Luetaan kirjastot
import pandas as pd
import os
import datetime
import numpy as np
import math
import shutil
from datetime import timedelta
import statsmodels.api as sm
import pickle
from sklearn import tree
pd.options.mode.chained_assignment = None 


data_path = 'set_your_path'

# read data file
# All variables are processed into data_file.pkl
with open('data_file.pkl', 'rb') as f:
    df_data = pickle.load(f)



#%%
# HELPER FUNCTIONS

def calc_desc_stats(df, variables_sel, variables_sel_names, titles_row, titles_name, df_column_names, variable_diff, group_column):
    import scipy.stats as stats

    variable_groups = []
    stats_all = []
    stats_x1 = []
    stats_x0 = []
    p_values = []

    variable_groups.append(' ')
    stats_all.append(len(df))
    stats_x1.append(len(df[df[variable_diff] == True]))
    stats_x0.append(len(df[df[variable_diff] == False]))
    p_values.append('')

    n = 0
    for k in range(len(variables_sel)):
        if df[variables_sel[k]].dtypes == 'bool':
            stats_all.append(str(int(df[variables_sel[k]].sum())) + ' (' + str(np.round(df[variables_sel[k]].mean() * 100, 2)) + '%)')
            stats_x0.append(str(int(df[df[variable_diff] == False][variables_sel[k]].sum())) + ' (' + str(np.round(df[df[variable_diff] == False][variables_sel[k]].mean() * 100, 2)) + '%)')
            stats_x1.append(str(int(df[df[variable_diff] == True][variables_sel[k]].sum())) + ' (' + str(np.round(df[df[variable_diff] == True][variables_sel[k]].mean() * 100, 2)) + '%)')
            variables_sel_names[k] = variables_sel_names[k] + ', n (%)'

            x0 = df[df[variable_diff] == False][variables_sel[k]]
            x1 = df[df[variable_diff] == True][variables_sel[k]]

            x0_t = x0.sum()
            x0_f = len(x0) - x0.sum()
            x1_t = x1.sum()
            x1_f = len(x1) - x1.sum()

            oddsratio, pvalue = stats.fisher_exact([[x0_t, x0_f], [x1_t, x1_f]])
            p_values.append(np.round(pvalue, 3))

        else:
            stats_all.append(str(np.round(df[variables_sel[k]].mean(), 2)) + ' (' + str(np.round(df[variables_sel[k]].std(), 2)) + ')')
            stats_x0.append(str(np.round(df[df[variable_diff] == False][variables_sel[k]].mean(), 2)) + ' (' + str(np.round(df[df[variable_diff] == False][variables_sel[k]].std(), 2)) + ')')
            stats_x1.append(str(np.round(df[df[variable_diff] == True][variables_sel[k]].mean(), 2)) + ' (' + str(np.round(df[df[variable_diff] == True][variables_sel[k]].std(), 2)) + ')')
            variables_sel_names[k] = variables_sel_names[k] + ', mean (std)'

            x0 = df[df[variable_diff] == False][variables_sel[k]]
            x1 = df[df[variable_diff] == True][variables_sel[k]]

            u_statistic, pVal = stats.mannwhitneyu(x0, x1)
            p_values.append(np.round(pVal, 3))

        if k == titles_row[n]:
            variable_groups.append(titles_name[n])
            if n < len(titles_row) - 1:
                n = n + 1
        else:
            variable_groups.append(' ')

    df_stats = pd.DataFrame()

    if group_column == True:
        df_stats['Group'] = variable_groups

    df_stats['Item name'] = ['Clients, n'] + variables_sel_names
    df_stats[df_column_names[0]] = stats_all
    df_stats[df_column_names[1]] = stats_x0
    df_stats[df_column_names[2]] = stats_x1
    df_stats['P-value'] = p_values

    return df_stats


def calc_desc_stats_clusters(df, variables_sel, variables_sel_names, df_column_name):
    stats_all=[]
    stats_all.append(len(df))
    for k in range(len(variables_sel)):
        if (df[variables_sel[k]].dtypes == 'bool'):
            stats_all.append(str(int(df[variables_sel[k]].sum())) + ' (' + str(np.round(df[variables_sel[k]].mean()*100,2)) + '%)')
        else:
            stats_all.append(str(np.round(df[variables_sel[k]].mean(),2)) + ' (' + str(np.round(df[variables_sel[k]].std(),2)) + ')')
    df_stats = pd.DataFrame()
    df_stats['Item name'] = ['Clients, n'] + variables_sel_names
    df_stats[df_column_name] = stats_all
    return df_stats



def rank_variables(X, Y, variables, variables_names, test_set_size, n_iterations):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    variables_best = []

    auc_all_uni = []
    coef_all_uni = []
    df_coefs = pd.DataFrame()

    for bb in range(n_iterations):
        # random train/test split (n random patients)
        n=test_set_size
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=n, random_state=bb, stratify=Y)

        # Univariate
        auc_uni = []
        auc_uni_random = []
        coef_uni = []
    
        for cc in range(len(variables)):
            x_train_uni = x_train[variables[cc]].reset_index()
            x_train_uni['index'] = 1
            x_test_uni = x_test[variables[cc]].reset_index()
            x_test_uni['index'] = 1
            clf = linear_model.LogisticRegression(random_state=bb, class_weight="balanced",solver='liblinear')
            clf.fit(x_train_uni, y_train)           
            y_pred_proba = clf.predict_proba(x_test_uni)
            auc_uni.append(roc_auc_score(y_test, y_pred_proba[:,1]))
            coef_uni.append(clf.coef_[0].tolist()[1])

        auc_all_uni.append(auc_uni)
        coef_all_uni.append(coef_uni)


    df_univariate_results = pd.DataFrame()
    df_univariate_results['code'] = variables
    df_variables_names = pd.DataFrame()
    df_variables_names['code'] = variables
    df_variables_names['names'] = variables_names
    names = []
    for k in range(len(df_univariate_results)):
        names.append(df_variables_names[df_variables_names['code']==df_univariate_results['code'][k]]['names'].reset_index(drop=True)[0])
    df_univariate_results['names'] = names

    df_aucs = pd.DataFrame()
    for dd in range(len(auc_all_uni)):
        df_aucs[str(dd)] = auc_all_uni[dd]
    df_univariate_results['AUC'] = df_aucs.mean(1).round(3)

    columns_temp = df_aucs.T.columns
    p_values = []
    for dd in range (len(columns_temp)):
        student_scores = df_aucs.T[columns_temp[dd]]
        t_stat, p_value = stats.ttest_1samp(student_scores, 0.5)
        p_values.append(np.round(p_value,3))

    df_coefs = pd.DataFrame()
    for dd in range(len(coef_all_uni)):
        df_coefs[str(dd)] = coef_all_uni[dd]
    df_univariate_results['coef'] = df_coefs.mean(1).round(4)


    ave_ci = []
    for aa in range(len(df_aucs)):
        ave, lb, ub = mean_confidence_interval(df_aucs.loc[aa].tolist(), confidence=0.95)
        ave_ci.append(str(np.round(ave,3)) + ' (' + str(np.round(lb,3)) + '-' + str(np.round(ub,3)) + ')')
    df_univariate_results['AUC, mean (CI95%)'] = ave_ci

    df_univariate_results['P values'] = p_values

    # 20 parasta univariate muuttujaa
    variables_best.append(df_univariate_results.sort_values(by=['AUC'],ascending=False).reset_index(drop=True)[:100]['code'].tolist())

    return variables_best, df_univariate_results


def mean_confidence_interval(data, confidence=0.95):
    import numpy as np
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h





#%%
var_sel = [  # set your variable names
'ika_leikkaus1',
'sukupuoli_f',
'astma',
'allergia',
'krooniset_keuhkosairaudet',
'mielenterveys',
'muistisairaudet',
'syopa',
'sydan_ja_verisuonet',
'lihavuus',
'diabetes',
'tules',
'Number_of_diseases',
'diag_CRSsNP',
'diag_CRSwNP',
'NERD',
'eosinofilia_before_baseline_ess',
'tablettikortisoni_before_baseline_ess',
'smoking',
'BaselineTotalEthmoid',
]

var_sel_names = [
'Age',
'Gender (female)',
'Asthma',
'Allergy',
'krooniset_keuhkosairaudet',
'mielenterveys',
'muistisairaudet',
'syopa',
'sydan_ja_verisuonet',
'lihavuus',
'diabetes',
'tules',
'Number_of_diseases',
'diag_CRSsNP',
'diag_CRSwNP',
'NERD',
'eosinofilia_before_baseline_ess',
'tablettikortisoni_before_baseline_ess',
'smoking',
'BaselineTotalEthmoid'
]

X = df_data[var_sel]

titles_row = [33,34]
titles_name = ['','']
df_column_names = ['All', 'Early relapse NO', 'Early relapse YES']
variable_diff = 'early_relapse_3_or_more_visits_6months'

# Table 1
df_stats = calc_desc_stats(X, var_sel.copy(), var_sel.copy(), titles_row, titles_name, df_column_names, variable_diff, False)

date = now.strftime('%Y%m%d')
df_stats.to_csv(data_path+'/table01_stats_'+date+'.csv')


#%%

#------------------
# Table 2
# RANK VARIABLES (AUC, linear regression)
y = X['early_relapse_3_or_more_visits_6months']
X_norm=(X[var_sel]-X[var_sel].mean())/X[var_sel].std()
variables_best, df_univariate_results = rank_variables(X_norm, y, var_sel, var_sel, test_set_size = 100, n_iterations = 10)
df_univariate_auc_sorted = df_univariate_results.sort_values(by=['AUC'],ascending=False).reset_index(drop=True)

df_univariate_auc_sorted.to_csv(data_path+'/table02_univariate_auc_'+date+'.csv')


#%%

#--------------------
# Table 3
# RANK VARIABLES (OR-VALUES)
odds_ratios_all = []
odds_ratios_all_l_ci = []
odds_ratios_all_u_ci = []
p_values = []

for aa in range (len(var_sel)):
    
    if (var_sel[aa] == 'ika_leikkaus1'):
        x = X[[var_sel[aa]] + ['sukupuoli_f']].astype(int)
    elif (var_sel[aa] == 'sukupuoli_f'):
        x = X[[var_sel[aa]] + ['ika_leikkaus1']].astype(int)
    else:
        x = X[[var_sel[aa]] + ['ika_leikkaus1','sukupuoli_f']].astype(int)
    
    #y = X['early_relapse_3_or_more_visits_knk_6months']
    y = X['early_relapse_3_or_more_visits_6months']
    
    x=(x-x.mean())/x.std()
    
    log_reg = sm.Logit(y,x, method='bfgs').fit()
    
    print(log_reg.summary())
    print(log_reg.params)
    odds_ratios = pd.DataFrame(
        {
            "OR": log_reg.params,
            "L_CI": log_reg.conf_int()[0],
            "U_CI": log_reg.conf_int()[1],
        }
    )
    
    odds = pd.DataFrame(np.exp(odds_ratios)).reset_index()[0:1]
    odds_ratios_all.append(np.round(odds['OR'].values[0],3))
    odds_ratios_all_l_ci.append(np.round(odds['L_CI'].values[0],3))
    odds_ratios_all_u_ci.append(np.round(odds['U_CI'].values[0],3))
    
    pvalue = np.round(log_reg.pvalues[0:1].values[0],3)
    low = False
    if (pvalue<0.001):
        pvalue = '<.001**'
        low = True
    else:
        pvalue = round(pvalue,3)
    if (low == False):
        if (pvalue<0.05):
            pvalue = str(pvalue)+'*'
    
    p_values.append(pvalue)
    
df_odds = pd.DataFrame()
df_odds['variables'] = var_sel
df_odds['OR'] = odds_ratios_all
df_odds['Lower CI 95%'] = odds_ratios_all_l_ci
df_odds['Upper CI 95%'] = odds_ratios_all_u_ci
df_odds['P-values'] = p_values
df_odds = df_odds.sort_values(by=['OR'], ascending=False).reset_index(drop=True)

df_odds.to_csv(data_path+'/table03_OR_'+date+'.csv')


#%%

#-----------------------
# Table 4 and Fig 1
# DECISION TREE CLUSTERING

y = X['early_relapse_3_or_more_visits_6months']

dt = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)
dt = dt.fit(X[var_sel], y)
#plt.figure(figsize=(100,20))
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,5), dpi=200)
tree.plot_tree(dt, feature_names = var_sel_names, impurity=False, proportion=False, fontsize=22, filled=False, label='all')
plt.savefig(data_path+'/fig_dt_clustering.tiff',bbox_inches = 'tight', facecolor='w')
plt.show()


X['cluster_dt']  = dt.apply(X[var_sel])

a=list(X['cluster_dt'].unique())
a.sort()
b = list(range(0, len(a)))
X['cluster_dt'] = X['cluster_dt'].replace(a, b)

df_stats_clusters_dt = pd.DataFrame()
# Patients all
df_stats = calc_desc_stats_clusters(X, var_sel_desc.copy(), var_sel_desc.copy(), 'All')
df_stats_clusters_dt['Item'] = df_stats['Item name']
df_stats_clusters_dt['All'] = df_stats['All']

for k in range(0,len(a)):
    df_temp = X[(X['cluster_dt']==k)][var_sel_desc].reset_index(drop=True)
    df_stats = calc_desc_stats_clusters(df_temp, var_sel_desc.copy(), var_sel_desc.copy(), str(k))
    df_stats_clusters_dt[str(k)] = df_stats[str(k)]
    
df_stats_clusters_dt.to_csv(data_path+'/table04_stats_clusters_dt_'+date+'.csv')