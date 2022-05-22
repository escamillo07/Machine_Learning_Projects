#!/usr/bin/env python
# coding: utf-8

# # Concrete Slump Test Regression

# The concrete slump test measures the consistency of fresh concrete before it sets. It is performed to check the workability of freshly made concrete, and therefore the ease with which concrete flows. It can also be used as an indicator of an improperly mixed batch.
# 
# <img src="https://i0.wp.com/civiconcepts.com/wp-content/uploads/2019/08/Slump-Cone-test-of-concrete.jpg?fit=977%2C488&ssl=1">
# 
# Our data set consists of various cement properties and the resulting slump test metrics in cm. Later on the set concrete is tested for its compressive strength 28 days later.
# 
# Input variables (9):
# 
# (component kg in one M^3 concrete)(7):
# * Cement
# * Slag
# * Fly ash
# * Water
# * SP
# * Coarse Aggr.
# * Fine Aggr.
# 
# (Measurements)(2)
# * SLUMP (cm)
# * FLOW (cm)
# 
# Target variable (1):
# * **28-day Compressive Strength (Mpa)**
# 
# Data Source: https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
# 
# *Credit: Yeh, I-Cheng, "Modeling slump flow of concrete using second-order regressions and artificial neural networks," Cement and Concrete Composites, Vol.29, No. 6, 474-480, 2007.*

# ## Dataset Description...(Basic Domain Kowledge about Slump Test)

# In[32]:


## The dataset consists of 103 instances with 10 attributes and has no missing values. There are 9 input variables and 1 output variable. Seven input variables represent the amount of raw material (measured in kg/m³) and one represents Age (in Days). The target variable is Concrete Compressive Strength measured in (MPa — Mega Pascal). We shall explore the data to see how input features are affecting compressive strength.
#  We can observe a high positive correlation between compressive Strength (CC_Strength) and Cement. this is true because strength concrete indeed increases with an increase in the amount of cement used in preparing it. Also, Age and Super Plasticizer are the other two factors influencing Compressive strength.
#  There are other strong correlations between the features,
#  A strong negative correlation between Super Plasticizer and Water.
#  positive correlations between Super Plasticizer and Fly Ash, Fine Aggregate.


# ## Input Variables
# 
# The Compressive Strength of Concrete determines the quality of Concrete. This is generally determined by a standard 
#     crushing test on a concrete cylinder. The recommended wait time for testing the cylinder is 28 days to ensure correct           results
#  
# Superplasticizers 
#     Superplasticizers (SPs), also known as high range water reducers, are additives used in making high strength concrete.
#     Plasticizers are chemical compounds that enable the production of concrete with approximately 15% less water content. 
#     Superplasticizers allow reduction in water content by 30% or more. These additives are employed at the level of a few weight percent. Plasticizers and superplasticizers retard the curing of concrete.
# 
# Flying ash: 
#     Fly ash is the finely divided residue that results from the combustion of pulverized coal and is transported from the 
#     combustion chamber by exhaust gases.
# 
# Coarse aggregate: 
#     Coarse aggregate is stone which are broken into small sizes and irregular in shape. In construction work the aggregate 
#     are used such as limestone and granite or river aggregate. Aggregate which has a size bigger than 4.75 mm or which 
#     retrained on 4.75 mm IS Sieve are known as Coarse aggregate.
# 
# Fine aggregates 
#     Fine aggregates are essentially any natural sand particles won from the land through the mining process. Fine aggregates 
#     consist of natural sand or any crushed stone particles that are ¼” or smaller. This product is often referred to as 1/4’” 
#     minus as it refers to the size, or grading, of this particular aggregate.
# 
# slump :
#     Concrete slump is a measurement of the workability or consistency of concrete.The concrete slump test measures the consistency
#     of fresh concrete before it sets. It is performed to check the workability of freshly made concrete, and therefore the ease 
#     with which concrete flows. It can also be used as an indicator of an improperly mixed batch. The test is popular due to the 
#     simplicity of apparatus used and simple procedure. The slump test is used to ensure uniformity for different loads of 
#     concrete under field conditions.
# 
# The flow of concrete: 
#     The percentage increase in the average diameter of the spreading concrete over the base diameter of the mould is called 
#     the flow of concrete.
# *\

# In[ ]:





# # Importing dependencies

# In[33]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# # Loading data

# In[34]:


df = pd.read_csv('cement_slump.csv')


# # EDA and Graphical analysis

# In[35]:


df.head()


# In[36]:


df.shape


# In[37]:


df.info()


# In[38]:


df.describe()


# In[39]:


sns.pairplot(df);


# In[40]:


df.corr()


# In[41]:


sns.heatmap(df.corr(), annot = True)


# In[45]:


plt.figure(figsize=(20, 8))

plt.subplot(121)
plt.hist(df["Compressive Strength (28-day)(Mpa)"], bins=50)

plt.subplot(122)
plt.boxplot(df["Compressive Strength (28-day)(Mpa)"], whis=2.5)

plt.show()


# In[46]:


# WHAT A WONDERFUL CODE....EXPLAİNS AND SHOWS EVERY DETAİL......
def plot_relation(df, target_var):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    c = 0
    for i in range(3):
        for j in range(3):
            if df.columns[c] != target_var:
                sns.regplot(x=df.columns[c],
                            y=target_var,
                            data=df,
                            ax=axs[i][j])
            c += 1
    plt.tight_layout()
plot_relation(df,"Compressive Strength (28-day)(Mpa)" )
scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()
train_val(y_train, y_train_pred, y_test, y_pred, "linear")


# # Data Preprocessing 

# ### Features and target variable

# In[47]:


X = df.drop(columns ="Compressive Strength (28-day)(Mpa)")
y = df["Compressive Strength (28-day)(Mpa)"]


# ### Splitting data into training and testing

# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)


# In[50]:


df.sample(15)


# In[51]:


print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


# ## Scaling

# In[52]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # Robustscaler is used when outlier could be present
scaler = StandardScaler()


# In[53]:


scaler.fit(X_train)


# In[54]:


X_train_scaled = scaler.transform(X_train) 
X_train_scaled


# In[55]:


X_test_scaled = scaler.transform(X_test)
X_test_scaled


# In[56]:


pd.DataFrame(X_train_scaled).agg(["mean", "std"]).round()


# In[57]:


pd.DataFrame(X_test_scaled).agg(["mean", "std"]).round()


# ##  1. Model Building (Linear Regression)

# In[86]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()


# In[87]:


lm.fit(X_train_scaled, y_train)


# In[88]:


y_pred = lm.predict(X_test_scaled)
y_train_pred = lm.predict(X_train_scaled)


# ### 1.1 Interpret the model

# In[89]:


lm.coef_


# In[90]:


lm.intercept_


# In[91]:


my_dict = {"Actual": y_test, "Pred": y_pred, "Residual":y_test-y_pred}


# In[92]:


comparing = pd.DataFrame(my_dict)
comparing


# In[93]:


result_sample = comparing.head(25)
result_sample


# In[94]:


result_sample.plot(kind ="bar", figsize=(15,9))
plt.show()


# ### 1.2 Model Evaluation

# In[95]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_val(y_train, y_train_pred, y_test, y_pred, name):
    
    scores = {name+"_train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    name+"_test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# In[96]:


ls =train_val(y_train, y_train_pred, y_test, y_pred, "linear")
ls


# ## <span style='color:red'> Multicolineratiy</span> 

# If there is a strong correlation between the independent variables, this situation is called **multicolineraty**.
#  
# **Multicolineraty** prevents my model from detecting important features.

# In[97]:


def color_red(val):
    if val > 0.90 and val < 0.99:
        color = 'red'
    else:
        color = 'black'
    return f'color: {color}'


# In[98]:


pd.DataFrame(df).corr().style.applymap(color_red)


# ## <span style='color:red'> Cross Validation</span> 

# We do cross-validation to check whether the one-time scores we receive are consistent or not
# 
# cross validation is only applied to the train set.

# ![image-2.png](attachment:image-2.png)

# In[105]:


from sklearn.model_selection import cross_validate


# In[106]:


model = LinearRegression()
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['r2', 'neg_mean_absolute_error','neg_mean_squared_error',                                                             'neg_root_mean_squared_error'], cv = 5)


# In[107]:


scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()


# In[108]:


train_val(y_train, y_train_pred, y_test, y_pred, "linear")


# # 2. Regularization

# ## 2.1 Ridge (Apply and evaluate)

# In[109]:


from sklearn.linear_model import Ridge


# In[110]:


ridge_model = Ridge(alpha=1, random_state=42)


# In[111]:


ridge_model.fit(X_train_scaled, y_train)


# In[112]:


y_pred = ridge_model.predict(X_test_scaled)
y_train_pred = ridge_model.predict(X_train_scaled)


# In[113]:


rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
rs


# In[114]:


pd.concat([ls, rs], axis=1)


# ## Choosing best alpha value with Cross-Validation

# In[115]:


from sklearn.linear_model import RidgeCV


# In[116]:


alpha_space = np.linspace(0.01, 1, 100)
alpha_space


# In[117]:


ridge_cv_model = RidgeCV(alphas=alpha_space, cv = 5, scoring= "neg_root_mean_squared_error")


# In[118]:


ridge_cv_model.fit(X_train_scaled, y_train)


# In[128]:


ridge_cv_model.alpha_ 


# In[129]:


ridge_cv_model.best_score_


# In[130]:


y_pred = ridge_cv_model.predict(X_test_scaled)
y_train_pred = ridge_cv_model.predict(X_train_scaled)


# In[131]:


rcs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge_cv")
rcs


# In[132]:


pd.concat([ls, rs, rcs], axis = 1)


# ## 2.2 Lasso (Apply and evalute)

# In[133]:


from sklearn.linear_model import Lasso, LassoCV


# In[134]:


lasso_model = Lasso(alpha=1, random_state=42)
lasso_model.fit(X_train_scaled, y_train)


# In[135]:


y_pred = lasso_model.predict(X_test_scaled)
y_train_pred = lasso_model.predict(X_train_scaled)


# In[136]:


lss = train_val(y_train, y_train_pred, y_test, y_pred, "lasso")
lss


# In[137]:


pd.concat([ls, rs, rcs, lss], axis = 1)


# ### Choosing best alpha value with Cross-Validation

# In[138]:


lasso_cv_model = LassoCV(alphas = alpha_space, cv = 5, max_iter=100000, random_state=42) 


# In[139]:


lasso_cv_model.fit(X_train_scaled, y_train)


# In[140]:


lasso_cv_model.alpha_


# In[141]:


np.where(alpha_space[::-1]==lasso_cv_model.alpha_)


# In[142]:


alpha_space[::-1]


# In[143]:


lasso_cv_model.mse_path_[99].mean()


# In[144]:


y_pred = lasso_cv_model.predict(X_test_scaled)  
y_train_pred = lasso_cv_model.predict(X_train_scaled)


# In[145]:


lcs = train_val(y_train, y_train_pred, y_test, y_pred, "lasso_cv")
lcs


# In[146]:


pd.concat([ls,rs, rcs, lss, lcs], axis = 1)


# In[147]:


lasso_cv_model.coef_


# ## 2.3 Elastic-Net (Apply and evaluate )
# * Use Gridsearch for hyperparameter tuning instead of ElacticnetCV

# In[148]:


from sklearn.linear_model import ElasticNet, ElasticNetCV


# In[149]:


elastic_model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42) # l1_ratio: 1: Lasso or 0:Ridge
elastic_model.fit(X_train_scaled, y_train)


# In[161]:


y_pred = elastic_model.predict(X_test_scaled)
y_train_pred = elastic_model.predict(X_train_scaled)


# In[162]:


es = train_val(y_train, y_train_pred, y_test, y_pred, "elastic")
es


# In[163]:


pd.concat([ls,rs,lss, es], axis = 1)


# ### Choosing best alpha and l1_ratio values with Cross-Validation

# In[164]:


elastic_cv_model = ElasticNetCV(alphas = alpha_space, l1_ratio=[0.1, 0.5, 0.7,0.9, 0.95, 1], cv = 5, 
                                max_iter = 100000,random_state=42)


# In[165]:


elastic_cv_model.fit(X_train_scaled, y_train)


# In[166]:


elastic_cv_model.alpha_


# In[167]:


elastic_cv_model.l1_ratio_


# In[168]:


y_pred = elastic_cv_model.predict(X_test_scaled)
y_train_pred = elastic_cv_model.predict(X_train_scaled)


# In[169]:


ecs = train_val(y_train, y_train_pred, y_test, y_pred, "elastic_cv")
ecs


# In[170]:


pd.concat([ls,rs, rcs, lss, lcs, es, ecs], axis = 1).T


# In[171]:


elastic_cv_model.coef_


# ## 2.4 Gridsearch

# In[173]:


from sklearn.model_selection import GridSearchCV


# In[174]:


elastic_model = ElasticNet(max_iter=10000, random_state=42)


# In[175]:


param_grid = {"alpha":[0.01, 0.012, 0.2, 0.5, 0.6, 0.7, 1],
            "l1_ratio":[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]}


# In[176]:


grid_model = GridSearchCV(estimator = elastic_model, param_grid = param_grid, scoring = 'neg_root_mean_squared_error',
                         cv =5, verbose =2)


# In[177]:


grid_model.fit(X_train_scaled, y_train)


# In[178]:


grid_model.best_params_


# In[179]:


pd.DataFrame(grid_model.cv_results_)


# In[180]:


grid_model.best_index_


# In[181]:


grid_model.best_score_


# ## Using Best Hyper Parameters From GridSearch

# In[182]:


y_pred = grid_model.predict(X_test_scaled)
y_train_pred = grid_model.predict(X_train_scaled)


# In[183]:


gs = train_val(y_train, y_train_pred, y_test, y_pred, "GridSearch")
gs


# In[184]:


pd.concat([ls,rs, rcs, lss, lcs, es, ecs, gs], axis = 1).T


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




