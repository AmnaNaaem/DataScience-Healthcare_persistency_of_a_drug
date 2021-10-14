#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.preprocessing as preprocessing 

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[2]:


data=pd.read_csv("datahealth.csv")


# In[3]:


df=data.copy()
df.head()


# # Basic Analysis

# In[4]:


df.shape


# In[5]:


df.info()


# # Data Cleansing and Transformation:

# Missing Values:

# In[6]:


in_df=df.columns[df.dtypes!='object']
obj_df=df.columns[df.dtypes=='object']


# In[7]:


print(in_df)


# In[8]:


print(obj_df)


# In[9]:


df[in_df].isnull().sum()


# In[10]:


df[obj_df].isnull().sum()


# Duplicate values:

# In[11]:


duplicate=df[df.duplicated()]
duplicate


# Outliers Detection and Removal:

# In[12]:


for x in df.columns[df.dtypes!=object]:
    fig=plt.figure()
    sns.boxplot(y=df[x],data=df)
    fig.suptitle(x)


# In[13]:


import warnings
warnings.filterwarnings('ignore')


# In[14]:


percentile25 = df['Dexa_Freq_During_Rx'].quantile(0.25)
percentile75 = df['Dexa_Freq_During_Rx'].quantile(0.75)
upper_limit = percentile75 + 1.5 * (percentile75 -percentile25 )
lower_limit = percentile25 - 1.5 * (percentile75 -percentile25 )
df[df['Dexa_Freq_During_Rx'] > upper_limit]
df[df['Dexa_Freq_During_Rx'] < lower_limit]
df3 = df[df['Dexa_Freq_During_Rx'] < upper_limit]
df3.shape


# In[15]:


df3_cap = df.copy()
df3_cap['Dexa_Freq_During_Rx'] = np.where(
    df3_cap['Dexa_Freq_During_Rx'] > upper_limit,
    upper_limit,
    np.where(
        df3_cap['Dexa_Freq_During_Rx'] < lower_limit,
        lower_limit,
        df3_cap['Dexa_Freq_During_Rx']
    )
)


# In[16]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.boxplot(df['Dexa_Freq_During_Rx'])
plt.subplot(2,2,2)
sns.boxplot(df3_cap['Dexa_Freq_During_Rx'])
plt.show()


# In[17]:


percentile25 = df3_cap['Count_Of_Risks'].quantile(0.25)
percentile75 = df3_cap['Count_Of_Risks'].quantile(0.75)
upper_limit = percentile75 + 1.5 * (percentile75 -percentile25 )
lower_limit = percentile25 - 1.5 * (percentile75 -percentile25 )
df3_cap[df3_cap['Count_Of_Risks'] > upper_limit]
df3_cap[df3_cap['Count_Of_Risks'] < lower_limit]
new_df = df3_cap[df3_cap['Count_Of_Risks'] < upper_limit]
new_df.shape


# In[18]:


new_df_cap = df3_cap.copy()
new_df_cap['Count_Of_Risks'] = np.where(
    new_df_cap['Count_Of_Risks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['Count_Of_Risks'] < lower_limit,
        lower_limit,
        new_df_cap['Count_Of_Risks']
    )
)


# In[19]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.boxplot(df3_cap['Count_Of_Risks'])
plt.subplot(2,2,2)
sns.boxplot(new_df_cap['Count_Of_Risks'])
plt.show()


# In[20]:


df=new_df_cap.copy()


# Grouping Sparse Classes:

# In[21]:


df1=df.drop(['Ptid'],axis=1)


# In[22]:


for x in df1.columns[df1.dtypes==object]:
    fig=plt.figure()
    df1[x].value_counts(normalize=True).plot(kind='barh')
    fig.suptitle(x)


# In[23]:


df['Ntm_Speciality'].value_counts()


# In[24]:


conditions=[
    (df['Ntm_Speciality'] == 'GENERAL PRACTITIONER'),
(df['Ntm_Speciality'] == 'RHEUMATOLOGY'),
(df['Ntm_Speciality'] == 'ENDOCRINOLOGY'),
(df['Ntm_Speciality'] == 'ONCOLOGY')
]


# In[25]:


choices=['GENERAL PRACTITIONER','RHEUMATOLOGY','ENDOCRINOLOGY','ONCOLOGY']


# In[26]:


df['Ntm_Speciality_Cat'] = np.select(conditions, choices, default='other')


# In[27]:


df['Ntm_Speciality_Cat'].value_counts()


# Categorical data encoding and computing correlation:

# In[28]:


from sklearn.preprocessing import LabelEncoder


# In[29]:


def number_encode_features(df): 
    result = df.copy()     
    encoders = {}     
    for column in result.columns:         
        if result.dtypes[column] == np.object:             
            encoders[column] = preprocessing.LabelEncoder() 
            result[column] = encoders[column].fit_transform(result[column]
) 
    return result, encoders  
# Calculate the correlation and plot it 
encoded_data, _ = number_encode_features(df)
encoded_data.drop(['Ptid'],axis=1).corr()


# In[30]:


def get_redundant_pairs(encoded_data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = encoded_data.columns
    for i in range(0, encoded_data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(encoded_data, n=5):
    au_corr = encoded_data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(encoded_data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(encoded_data, 3))


# # Exploratory Data Analysis:

# In[31]:


for x in df1.columns[df1.dtypes==object]:
    fig=plt.figure()
    sns.countplot(y=df1[x],hue=df1['Persistency_Flag'],data=df1)
    fig.suptitle(x)


# In[32]:


df.groupby('Persistency_Flag')['Gender'].value_counts(normalize=True).plot.pie(autopct="%.1f%%");


# In[33]:


data=df.groupby('Persistency_Flag')['Gender'].value_counts(normalize=True)


# In[34]:


pie, ax = plt.subplots(figsize=[10,6])
labels = data.keys()
plt.pie(x=data, autopct="%.1f%%", explode=[0.05]*4, labels=labels, pctdistance=0.5)


# In[35]:


df.groupby('Persistency_Flag')['Ethnicity'].value_counts(normalize=True).plot.barh()


# In[36]:


df.groupby('Persistency_Flag')['Region'].value_counts(normalize=True).plot.barh()


# In[37]:


fig_dim=(8,5)
fig, ax= plt.subplots(figsize=fig_dim)
df.groupby('Persistency_Flag')['Region'].value_counts(normalize=True).unstack('Region').plot.barh(stacked=True,ax=ax);
plt.legend(bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)


# In[38]:


p1=df.groupby(['Persistency_Flag','Gender'])['Count_Of_Risks'].sum().reset_index()


# In[39]:


p1


# In[40]:


sns.catplot(y="Age_Bucket",hue="Persistency_Flag", kind="count",
            palette="pastel", edgecolor=".6",
            data=df)


# In[41]:


sns.catplot(y="Ntm_Speciality_Cat",hue="Persistency_Flag", kind="count",
            palette="pastel", edgecolor=".6",
            data=df)


# In[42]:


fig_dim=(8,5)
fig, ax= plt.subplots(figsize=fig_dim)
df.groupby(['Persistency_Flag','Ntm_Speciality_Cat'])['Region'].value_counts(normalize=True).unstack('Region').plot.barh(stacked=True,ax=ax);
plt.legend(bbox_to_anchor=(1.02,1),loc='upper left', borderaxespad=0)


# In[43]:


risk_cols = [col for col in df.columns if 'Risk' in col]
for x in risk_cols:
    fig=plt.figure()
    sns.countplot(y=df[x],hue=df['Persistency_Flag'],data=df)
    fig.suptitle(x)


# In[44]:


ix = 1
fig = plt.figure(figsize = (15,10))
for c in list(risk_cols):
    if ix <= 3:
        ax1 = fig.add_subplot(2,3,ix)
        sns.countplot(data = df, y=df[x],hue="Persistency_Flag", ax = ax1)
        ax2 = fig.add_subplot(2,3,ix+3)
        sns.boxplot(data=df, y=df[x], ax=ax2)
            #sns.violinplot(data=ds_cat, x=c, y='SalePrice', ax=ax2)
            #sns.swarmplot(data = ds_cat, x=c, y ='SalePrice', color = 'k', alpha = 0.4, ax=ax2)
        fig.suptitle(x)   
    ix = ix +1
    if ix == 4: 
        fig = plt.figure(figsize = (15,10))
        ix =1


# # Feature Selection and Modeling:

# In[45]:


df2=encoded_data.drop(['Ptid','Dexa_Freq_During_Rx','Ntm_Speciality','Risk_Segment_Prior_Ntm'],axis=1)


# In[46]:


df2.head()


# In[47]:


X = df2.drop(['Persistency_Flag'], axis=1)
Y = df2['Persistency_Flag']


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.3, random_state=25, shuffle=True)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


lr_model = LogisticRegression(random_state=25)


# In[52]:


lr_model.fit(xtrain, ytrain)


# In[53]:


pred = lr_model.predict(xtest)


# In[54]:


pred[0:9]


# In[55]:


pred_prb = lr_model.predict_proba(xtest)


# In[56]:


pred_prb[0:9, 0:9]


# In[57]:


lr_pred_prb = lr_model.predict_proba(xtest)[:, 1]


# In[58]:


xtest.head()


# In[59]:


xt = xtest.copy()
xt['pred'] = pred
xt['pred_probability'] = lr_pred_prb
xt['actual'] = ytest
xt.head()


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


confusion_matrix(ytest, pred)


# In[62]:


confusion_matrix(ytest, pred).ravel()


# In[63]:


tn, fp, fn, tp = confusion_matrix(ytest, pred).ravel()
conf_matrix = pd.DataFrame({"pred_Persistent": [tp, fp], "pred_Non-Persistent": [
                           fn, tn]}, index=["Persistent", "Not Persistent"])
conf_matrix


# In[64]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[65]:


precision_lr = precision_score(ytest, pred)
print("Precision by built-in function: {}".format(precision_lr))
recall_lr = recall_score(ytest, pred)
print("Recall by built-in function: {}".format(recall_lr))
accuracy_lr = accuracy_score(ytest, pred)
print("Accuracy by built-in function: {}".format(accuracy_lr))
f1_lr = f1_score(ytest, pred)
print("F1 Score by built-in function: {}".format(f1_lr))


# In[66]:


tpr = recall_lr
fpr = fp / (fp + tn)


# In[67]:


tpr, fpr


# In[68]:


fpr = 1 - recall_lr
tpr, fpr


# In[69]:


from sklearn.metrics import auc, roc_curve, roc_auc_score


# In[70]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')


# In[71]:


fpr, tpr, threshold = roc_curve(ytest, lr_pred_prb)


# In[72]:


auc_lr = roc_auc_score(ytest, lr_pred_prb)
auc_lr


# In[73]:


sns.set_context('poster')
plot_roc_curve(fpr, tpr, label='AUC = %0.3f' % auc_lr)


# In[74]:


from sklearn.tree import DecisionTreeClassifier


# In[75]:


clf_dt = DecisionTreeClassifier(
    max_depth=3, criterion='gini', random_state=100)


# In[76]:


clf_dt.fit(xtrain, ytrain)


# In[77]:


dt_pred = clf_dt.predict(xtest)
dt_pred_prb = clf_dt.predict_proba(xtest)[:, 1]


# In[78]:


accuracy_dt = accuracy_score(ytest,dt_pred)
print("Accuracy: {}".format(accuracy_dt))
precision_dt=precision_score(ytest,dt_pred)
print("Precision: {}".format(precision_dt))
recall_dt = recall_score(ytest,dt_pred)
print("Recall: {}".format(recall_dt))
dt_f1=f1_score(ytest,dt_pred)
print("F1 Score: {}".format(dt_f1))


# In[79]:


sns.set_context('poster')
auc_dt = roc_auc_score(ytest, dt_pred_prb)
fpr, tpr, threshold = roc_curve(ytest, dt_pred_prb)
plot_roc_curve(fpr, tpr, label='AUC = %0.3f' % auc_dt)


# In[80]:


from sklearn.ensemble import RandomForestClassifier


# In[81]:


clf_rf = RandomForestClassifier(random_state=100)


# In[82]:


clf_rf.fit(xtrain, ytrain)


# In[83]:


rf_pred = clf_rf.predict(xtest)
rf_pred_prb = clf_rf.predict_proba(xtest)[:, 1]


# In[84]:


precision_rf=precision_score(ytest,rf_pred)
print("Precision: {}".format(precision_rf))
accuracy_rf = accuracy_score(ytest,rf_pred)
print("Accuracy: {}".format(accuracy_rf))
recall_rf = recall_score(ytest,rf_pred)
print("Recall: {}".format(recall_rf))
rf_f1=f1_score(ytest,rf_pred)
print("F1 Score: {}".format(rf_f1))


# In[85]:


auc_rf = roc_auc_score(ytest, rf_pred_prb)
fpr, tpr, threshold = roc_curve(ytest, rf_pred_prb)
plot_roc_curve(fpr, tpr, label='AUC = %0.3f' % auc_rf)


# In[86]:


from sklearn.ensemble import AdaBoostClassifier


# In[87]:


clf_adb = AdaBoostClassifier(random_state=100)
clf_adb.fit(xtrain, ytrain)


# In[88]:


pred_clf_adb = clf_adb.predict(xtest)


# In[89]:


adb_pred_prb = clf_adb.predict_proba(xtest)[:, 1]


# In[90]:


accuracy_adb=accuracy_score(ytest,pred_clf_adb)
print("Accuracy: {}".format(accuracy_adb))
precision_adb=precision_score(ytest, pred_clf_adb)
print("Precision: {}".format(precision_adb))
recall_adb=recall_score(ytest,pred_clf_adb)
print("Recall_score: {}".format(recall_adb))
f1_adb=f1_score(ytest, pred_clf_adb)
print("F1 score: {}".format(f1_adb))


# In[91]:


auc_adb = roc_auc_score(ytest, adb_pred_prb)
fpr, tpr, threshold = roc_curve(ytest, adb_pred_prb)
plot_roc_curve(fpr, tpr, label='AUC = %0.3f' % auc_adb)


# In[92]:


import xgboost as xgb


# In[93]:


clf_xgb = xgb.XGBClassifier(seed=25,nthread=1,random_state=100)


# In[94]:


clf_xgb.fit(xtrain, ytrain)


# In[95]:


xgb_pred = clf_xgb.predict(xtest)
xgb_pred_prb=clf_xgb.predict_proba(xtest)[:,1]


# In[96]:


accuracy_xgb = accuracy_score(ytest,xgb_pred)
print("Accuracy: {}".format(accuracy_xgb))
recall_xgb = recall_score(ytest,xgb_pred)
print("Recall: {}".format(recall_xgb))
precision_xgb = precision_score(ytest,xgb_pred)
print("Precision: {}".format(precision_xgb))
f1_xgb=f1_score(ytest, xgb_pred)
print("F1 score: {}".format(f1_xgb))


# In[97]:


auc_xgb=roc_auc_score(ytest,xgb_pred_prb)
fpr,tpr,threshold=roc_curve(ytest,xgb_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_xgb)


# In[98]:


comparison_dict={"Algorithm":["Logistic Regression","Decision Tree","Random Forest","XGBoost","Ada Boost"],
                 "Accuracy":[accuracy_lr,accuracy_dt,accuracy_rf,accuracy_xgb,accuracy_adb],
                 "Precision":[precision_lr,precision_dt,precision_rf,precision_xgb,precision_adb],
                 "Recall":[recall_lr,recall_dt,recall_rf,recall_xgb,recall_adb],
                 "AUC":[auc_lr,auc_dt,auc_rf,auc_xgb,auc_adb],
                 "F1 Score":[f1_lr,dt_f1,rf_f1,f1_xgb,f1_adb]
                }


# In[100]:


comparison = pd.DataFrame(comparison_dict)
comparison.sort_values(['Recall', 'Accuracy', 'AUC'])


# In[ ]:




