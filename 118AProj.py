#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss #cross-entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV #platt scaling/isotonic regression
import pandas as pd


# In[2]:


adult_main = pd.read_csv("adult_dat.data")
adult_test = pd.read_csv("adult_test.test")
letters_dat = pd.read_csv("letter-recognition.data")
cov_dat = pd.read_csv("covtype.data")


# # Adult Data Processing

# In[3]:


adult = pd.concat([adult_main,adult_test], ignore_index = True)
adult.isnull().values.any()


# In[4]:


work = pd.get_dummies(adult['workclass']) ##
work.rename(columns = {' ?': 'NA_Work'}, inplace = True)
education = pd.get_dummies(adult['education'])
marital = pd.get_dummies(adult['marital-status'])
occu = pd.get_dummies(adult['occupation']) ##
occu.rename(columns = {' ?': 'NA_Occu'}, inplace = True)
relat = pd.get_dummies(adult['relationship'])
race = pd.get_dummies(adult['race'])
sex = pd.get_dummies(adult['sex'])
country = pd.get_dummies(adult['native-country']) ##
country.rename(columns = {' ?': 'NA_Country'}, inplace = True)


# In[5]:


adult_hot= pd.concat([work,education,marital,occu,relat,race,sex,country], axis = 1)


# In[6]:


adult_hot = pd.concat([adult[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']],adult_hot], axis =1)
adult['income'] = adult['income'].replace([' >50K.'], ' >50K')
adult['income'] = adult['income'].replace([' <=50K.'], " <=50K")


# In[7]:


income_binary = (adult['income'] == ' >50K').astype(int) 
adult_hot = pd.concat([adult_hot,income_binary],axis = 1)
adult_hot = adult_hot[adult_hot.NA_Work == 0]
adult_hot = adult_hot[adult_hot.NA_Occu == 0]
adult_hot = adult_hot[adult_hot.NA_Country == 0]


# In[8]:


adult_hot.shape


# In[9]:


adult_hot.drop(['NA_Work','NA_Occu','NA_Country'],axis = 1, inplace=True)


# In[10]:


adult_hot.shape


# In[11]:


adult_hot.reset_index(inplace = True)


# In[88]:


len(adult_hot.columns)


# # Cov Data Processing

# In[12]:


for i in range(1,8,1):
  print(sum((cov_dat['cov']==i).astype(int))) ##check which class is largest = class 2


# In[13]:


class2 = pd.DataFrame((cov_dat['cov'] == 2).astype(int))
class2.rename(columns = {'cov':'class2'}, inplace = True)


# In[14]:


del cov_dat['cov']
cov_bin = pd.concat([cov_dat,class2], axis = 1)


# In[15]:


cov_bin.isnull().any(); #check nullity


# # Letter Data Processing 

# In[17]:


switch = letters_dat['lettr']
del letters_dat['lettr']
letters_dat = pd.concat([letters_dat,switch],1)


# In[18]:


O_pos = pd.DataFrame((letters_dat['lettr'] == 'O').astype(int))
O_pos = O_pos.rename(columns = {'lettr':'O_pos'})


# In[19]:


letters_dat = pd.concat([letters_dat,O_pos],axis = 1)


# In[20]:


AM_pos = pd.DataFrame((letters_dat['lettr'] == 'A').astype(int))
for i in ['B','C','D','E','F','G','H','I','J','K','L','M']:
      AM_pos = AM_pos + pd.DataFrame((letters_dat['lettr'] == i).astype(int))


# In[21]:


AM_pos = AM_pos.rename(columns = {'lettr':'AM_pos'})


# In[22]:


letters_dat = pd.concat([letters_dat,AM_pos], axis = 1)


# In[23]:


del letters_dat['Unnamed: 17']


# In[24]:


letters_dat.shape


# ### Function for plotting ROC/Precision-Recall with KNN/Logit

# In[25]:


##different than SVM since we have to use best_fit.decision_function(..)
##predict_proba with SVM is both computationally expensive and doesn't give consistent results


# In[26]:


def roc_PR_plot(results, dataset_string, classifier_string):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
    tru_ys = results.get('y_true')
    probabilities_pos = results.get('tst_prob').loc[:,1]
    fp1, tp1, thresholds1 = roc_curve(tru_ys.get('y1'), probabilities_pos.iloc[:,0])
    fp2, tp2 , thresholds2 = roc_curve(tru_ys.get('y2'), probabilities_pos.iloc[:,1])
    fp3, tp3 , thresholds3 = roc_curve(tru_ys.get('y3'), probabilities_pos.iloc[:,2])
    ax[0].plot([0, 1], [0, 1], color='blue', linestyle='--', label = "diagonal")
    ax[0].plot(fp1, tp1, color='orange', label='ROC1')
    ax[0].plot(fp2, tp2, color='red', label='ROC2')
    ax[0].plot(fp3, tp3, color='green', label='ROC3')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC ' + classifier_string + ' over 3 trials, Dataset = ' + dataset_string)
    ax[0].legend()
    f1, t1, thres1 = precision_recall_curve(tru_ys.get('y1'), probabilities_pos.iloc[:,0])
    f2, t2 , thresh2 = precision_recall_curve(tru_ys.get('y2'), probabilities_pos.iloc[:,1])
    f3, t3 , thresholds3 = precision_recall_curve(tru_ys.get('y3'), probabilities_pos.iloc[:,2])
    ax[1].plot([0, 1], [1,0], color='blue', linestyle='--', label = "diagonal")
    ax[1].plot(f1, t1, color='orange', label='PR1')
    ax[1].plot(f2, t2, color='red', label='PR2')
    ax[1].plot(f3, t3, color='green', label='PR3')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall ' + classifier_string + ' over 3 trials, Dataset = ' + dataset_string)
    ax[1].legend()
    plt.show()


# # Begin KNN Trials

# In[75]:


def KNN(dataset, target,lower,upper,test_size, scaler):
    grid = np.arange(1,500,20)
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    tst_prob = pd.DataFrame()
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    roc_scores_1 = np.zeros(3)
    f1_scores = np.zeros(3)


    for j in [1,2,3]:
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.iloc[:,lower:upper], dataset.loc[:,target], test_size= test_size, random_state = j, shuffle = True)
        sca = scaler
        X_tr = sca.fit_transform(X_tr) 
        X_tst = sca.transform(X_tst)
        five_fold = KFold(n_splits = 5, shuffle = True)
        knn = KNeighborsClassifier()
        parameters = {'n_neighbors': list(grid), 'weights': ['uniform','distance']}
        
        print(X_tr.shape[0]) #print statement for progress
        GS = GridSearchCV(estimator = knn, param_grid = parameters, cv = five_fold, n_jobs = 4, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        validation_scores[j-1]= best_fit.best_score_
        print(best_fit.best_score_) #print statement for progress
        predictions = best_fit.predict(X_tst)
        prediction_prob = best_fit.predict_proba(X_tst)
        
        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(j)])], axis = 1)
        tst_prob = pd.concat([tst_prob,pd.DataFrame(prediction_prob).reset_index(drop = True)], axis = 1)
        params_best['best' + str(j)]= best_fit.best_params_
        y_true['y' + str(j)] = y_tst
        acc_scores[j-1] = accuracy_score(y_tst, predictions)
        precision_scores[j-1] = precision_score(y_tst, predictions)
        recall_scores[j-1] = recall_score(y_tst, predictions)
        roc_scores_1[j-1] = roc_auc_score(y_tst, prediction_prob[:,1])
        f1_scores[j-1] = f1_score(y_tst,predictions)
    
    dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions, 'tst_prob': tst_prob, 'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'roc_scores': roc_scores_1, 'f1_scores': f1_scores}
        
    
        
       
    return dict_of_vals
    


# ## Letter Dat

# In[76]:


results = KNN(letters_dat,'O_pos',0,16,.75, StandardScaler()) #unbalanced


# In[29]:


roc_PR_plot(results, 'Letter_OPos', 'KNN')


# In[30]:


results_AM = KNN(letters_dat,'AM_pos',0,16,.75, StandardScaler()) #balanced


# In[32]:


roc_PR_plot(results_AM, 'Letter_AM', 'KNN')


# ## Cov Dat

# In[28]:


results_cov = KNN(cov_bin, 'class2', 0,54, 576012, StandardScaler())


# In[30]:


roc_PR_plot(results_cov, 'COV', 'KNN')


# ## Adult Dat

# In[31]:


results_adult = KNN(adult_hot, 'income', 0, 105, 40222, StandardScaler())


# In[32]:


roc_PR_plot(results_adult, 'ADULT', 'KNN')


# # Begin SVM Trials

# In[78]:


def roc_PR_plot_SVM(results, dataset_string, classifier_string):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
    tru_ys = results.get('y_true')
    probabilities_pos = results.get('tst_prob')
    fp1, tp1, thresholds1 = roc_curve(tru_ys.get('y1'), probabilities_pos.iloc[:,0])
    fp2, tp2 , thresholds2 = roc_curve(tru_ys.get('y2'), probabilities_pos.iloc[:,1])
    fp3, tp3 , thresholds3 = roc_curve(tru_ys.get('y3'), probabilities_pos.iloc[:,2])
    ax[0].plot([0, 1], [0, 1], color='blue', linestyle='--', label = "diagonal")
    ax[0].plot(fp1, tp1, color='orange', label='ROC1')
    ax[0].plot(fp2, tp2, color='red', label='ROC2')
    ax[0].plot(fp3, tp3, color='green', label='ROC3')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC ' + classifier_string + ' over 3 trials, Dataset = ' + dataset_string)
    ax[0].legend()
    f1, t1, thres1 = precision_recall_curve(tru_ys.get('y1'), probabilities_pos.iloc[:,0])
    f2, t2, thresh2 = precision_recall_curve(tru_ys.get('y2'), probabilities_pos.iloc[:,1])
    f3, t3 , thres3 = precision_recall_curve(tru_ys.get('y3'), probabilities_pos.iloc[:,2])
    ax[1].plot([0, 1], [1,0], color='blue', linestyle='--', label = "diagonal")
    ax[1].plot(f1, t1, color='orange', label='PR1')
    ax[1].plot(f2, t2, color='red', label='PR2')
    ax[1].plot(f3, t3, color='green', label='PR3')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall ' + classifier_string + ' over 3 trials, Dataset = ' + dataset_string)
    ax[1].legend()
    plt.show()


# In[58]:


def SVM(dataset, target,lower,upper,test_size, scaler):
    
    import re
    
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    tst_prob = pd.DataFrame()
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    roc_scores_1 = np.zeros(3)
    f1_scores = np.zeros(3)

    
    def unpack_more(params):
        new_dict = {}
        test_dict = params
        for key in test_dict.keys() : 
            new_dict[re.sub('classifier__', '', key)] = (test_dict[key])
        return new_dict 
    


    for j in [1,2,3]:
        
        pipe = Pipeline([('std', scaler), ('classifier', SVC())])
        param_grid = [{'classifier__kernel': ['rbf'], 'classifier__C': np.power(10., np.arange(-7, 4)), 
                       'classifier__gamma':[0.001,0.005,0.01,0.05,0.1,0.5,1,2]},
                  {'classifier__kernel': ['poly'], 'classifier__C': np.power(10., np.arange(-7, 4)), 'classifier__degree': [2,3]},
                    {'classifier__kernel': ['linear'], 'classifier__C': np.power(10., np.arange(-7, 4))}]
        
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.iloc[:,lower:upper], dataset.loc[:,target], test_size= test_size, random_state = j, shuffle = True)
        five_fold = KFold(n_splits = 5, shuffle = True)
        
        print(X_tr.shape[0]) ##print statement to track progress
        GS = GridSearchCV(estimator = pipe, param_grid = param_grid, cv = five_fold, n_jobs = -1, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        print(best_fit.best_score_) ##print statement to track progress
        validation_scores[j-1]= best_fit.best_score_
        predictions = best_fit.predict(X_tst)
        decision = best_fit.decision_function(X_tst)
        prediction_prob = np.exp(decision)/(np.exp(decision)+1) ##bad practice
        
        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(j)])], axis = 1)
        tst_prob = pd.concat([tst_prob,pd.DataFrame(prediction_prob).reset_index(drop = True)], axis = 1)
        params_best['best' + str(j)]= best_fit.best_params_
        y_true['y' + str(j)] = y_tst
        acc_scores[j-1] = accuracy_score(y_tst, predictions)
        precision_scores[j-1] = precision_score(y_tst, predictions)
        recall_scores[j-1] = recall_score(y_tst, predictions)
        roc_scores_1[j-1] = roc_auc_score(y_tst, prediction_prob)
        f1_scores[j-1] = f1_score(y_tst,predictions)
    
    dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions, 'tst_prob': tst_prob, 'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'roc_scores': roc_scores_1, 'f1_scores': f1_scores}
        
    
        
       
    return dict_of_vals


# ## Letter Data

# In[59]:


letter_SVM = SVM(letters_dat, 'O_pos', 0,16,.75, StandardScaler())


# In[81]:


roc_PR_plot_SVM(letter_SVM, 'Letter_Opos', 'SVM')


# In[62]:


letter_SVM_AM = SVM(letters_dat, 'AM_pos', 0,16,.75, StandardScaler())


# In[69]:


AM_svm_val = np.array([0.9602, 0.9572, 0.9542])
AM_svm_acc = np.array([0.9624    , 0.96606667, 0.96606667])
AM_svm_prec = np.array([0.96103198, 0.96797297, 0.96727273])
AM_svm_recall = np.array([0.96310261, 0.96341627, 0.96428571])
AM_svm_roc = np.array([0.99276918, 0.99314926, 0.993619  ])
AM_svm_f1 = np.array([0.96206618, 0.96568925, 0.96577691])


# In[82]:


roc_PR_plot_SVM(letter_SVM_AM, 'Letter_AMPos', 'SVM')


# ## Adult Data

# In[63]:


adult_svm = SVM(adult_hot, 'income', 0, 105, 40222, StandardScaler())


# In[83]:


roc_PR_plot_SVM(adult_svm, 'ADULT', 'SVM')


# ## Cov Data

# In[65]:


cov_svm = SVM(cov_bin, 'class2', 0,54, 576012, StandardScaler())


# In[84]:


roc_PR_plot_SVM(cov_svm, 'COV', 'SVM')


# # Begin Logit Trials

# In[69]:


def Logistic(dataset, target,lower,upper,test_size,scaler):
    
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    tst_prob = pd.DataFrame()
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    roc_scores_1 = np.zeros(3)
    f1_scores = np.zeros(3)
    

    for j in [1,2,3]:
        
        pipe = Pipeline([('std', scaler), ('classifier', LogisticRegression(solver = 'lbfgs', max_iter = 1000))])
        param_grid = [{'classifier__penalty': ['l2'], 'classifier__C': np.power(10., np.arange(-8, 5))},
                {'classifier__penalty': ['none'],'classifier__C': np.power(10., np.arange(-8, 5))}]

    
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.iloc[:,lower:upper], dataset.loc[:,target], test_size= test_size, random_state = j, shuffle = True)
        five_fold = KFold(n_splits = 5, shuffle = True)
        
        print(X_tr.shape[0]) #print statement to check progress
        GS = GridSearchCV(estimator = pipe, param_grid = param_grid, cv = five_fold, n_jobs = 4, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        validation_scores[j-1]= best_fit.best_score_
        print(best_fit.best_score_) #print statement to check progress
        predictions = best_fit.predict(X_tst)
        prediction_prob = best_fit.predict_proba(X_tst)
        
        
        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(j)])], axis = 1)
        tst_prob = pd.concat([tst_prob,pd.DataFrame(prediction_prob).reset_index(drop = True)], axis = 1)
        params_best['best' + str(j)]= best_fit.best_params_
        y_true['y' + str(j)] = y_tst
        acc_scores[j-1] = accuracy_score(y_tst, predictions)
        precision_scores[j-1] = precision_score(y_tst, predictions)
        recall_scores[j-1] = recall_score(y_tst, predictions)
        roc_scores_1[j-1] = roc_auc_score(y_tst, prediction_prob[:,1])
        f1_scores[j-1] = f1_score(y_tst,predictions)
    
    dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions, 'tst_prob': tst_prob, 'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'roc_scores': roc_scores_1, 'f1_scores': f1_scores}
        
     
        
       
    return dict_of_vals


# In[70]:


results_O_log = Logistic(letters_dat,'O_pos',0,16,.75, StandardScaler()) #unbalanced


# In[72]:


roc_PR_plot_SVM(results_O_log, 'Letter', 'logit')


# In[73]:


results_AM_log = Logistic(letters_dat,'AM_pos',0,16,.75, StandardScaler()) #balanced


# In[75]:


roc_PR_plot_SVM(results_AM_log, 'Letter', 'logit')


# In[76]:


cov_log= Logistic(cov_bin, 'class2', 0,54, 576012, StandardScaler())


# In[78]:


roc_PR_plot_SVM(cov_log, 'Cov', 'logit')


# In[79]:


adult_log = Logistic(adult_hot, 'income', 0, 105, 40222, StandardScaler())


# In[81]:


roc_PR_plot_SVM(adult_log, 'Adult', 'logit')


# ## ALL RESULTS.
# ### --had to retrieve them manully from the returned dictionary since jupyter lab shut down

# In[2]:


##Need O_pos KNN
## Need all results from SVM

O_knn_val = np.array([0.9884, 0.9888, 0.9906])
O_knn_acc = np.array([0.99133333, 0.99026667, 0.99093333])
O_knn_prec = np.array([0.89630931, 0.84124386, 0.88766114])
O_knn_recall = np.array([0.8777969 , 0.91296625, 0.86535009])
O_knn_roc = np.array([0.93685254, 0.9531237 , 0.9305633 ])
O_knn_f1 = np.array([0.88695652, 0.87563884, 0.87636364])

cov_knn_val = np.array([0.7644, 0.771 , 0.7616])
cov_knn_acc = np.array([0.77929279, 0.76678784, 0.77316792])
cov_knn_prec = np.array([0.76566599, 0.75334624, 0.76028943])
cov_knn_recall = np.array([0.78872482, 0.77579572, 0.7809207 ])
cov_knn_roc = np.array([0.77952156, 0.84640168, 0.77335698])
cov_knn_f1 = np.array([0.77702437, 0.76440619, 0.77046698])

adult_knn_val = np.array([0.8272, 0.8346, 0.8278])
adult_knn_acc = np.array([0.82442444, 0.82474765, 0.82526975])
adult_knn_prec = np.array([0.71873596, 0.70062112, 0.73140367])
adult_knn_recall = np.array([0.48071722, 0.50984331, 0.46990602])
adult_knn_roc = np.array([0.8749578 , 0.87187256, 0.87591739])
adult_knn_f1 = np.array([0.57611044, 0.59019824, 0.57219382])

AM_knn_val = np.array([0.9468, 0.943 , 0.944 ])
AM_knn_acc = np.array([0.9546    , 0.95346667, 0.9536])
AM_knn_prec = np.array([0.95177495, 0.95087672, 0.95254692])
AM_knn_recall = np.array([0.9567735 , 0.95548083, 0.95408163])
AM_knn_roc = np.array([0.95462124, 0.95348397, 0.95360332])
AM_knn_f1 = np.array([0.95426768, 0.95317322, 0.95331366])

cov_log_val = np.array([0.7592, 0.7616, 0.7532])
cov_log_acc = np.array([0.75468567, 0.75341312, 0.75970987])
cov_log_prec = np.array([0.73840708, 0.74472044, 0.74148755])
cov_log_recall = np.array([0.7694581 , 0.75221513, 0.77852404])
cov_log_roc = np.array([0.82486558, 0.82433487, 0.82397922])
cov_log_f1 = np.array([0.75361288, 0.74844903, 0.75955458])


AM_log_val = np.array([0.736 , 0.7206, 0.731 ])
AM_log_acc = np.array([0.7262, 0.7258, 0.727 ])
AM_log_prec = np.array([0.71240241, 0.7159948 , 0.71685422])
AM_log_recall = np.array([0.74952868, 0.74055145, 0.74409237])
AM_log_roc = np.array([0.81066856, 0.8131976 , 0.8124264 ])
AM_log_f1 =  np.array([0.73049413, 0.72806612, 0.73021938])

adult_log_val = np.array([0.841 , 0.8466, 0.8418])
adult_log_acc = np.array([0.84456268, 0.84391626, 0.84289692])
adult_log_prec = np.array([0.72853118, 0.7222088 , 0.72681365])
adult_log_recall = np.array([0.59571271, 0.6003415 , 0.589982  ])
adult_log_roc = np.array([0.89477575, 0.89843187, 0.89665055])
adult_log_f1 = np.array([0.65546126, 0.65566038, 0.65128856])

O_log_val = np.array([0.9656, 0.962 , 0.9608])
O_log_acc = np.array([0.96126667, 0.96246667, 0.96286667])
O_log_prec = np.array([0,0,0])
O_log_recall = np.array([0,0,0])
O_log_roc = np.array([0.82614902, 0.83762008, 0.84356906])
O_log_f1 = np.array([0., 0., 0.])

cov_svm_val = np.array([0.799 , 0.7926, 0.7862])
cov_svm_acc = np.array([0.80562558, 0.78864329, 0.8025805 ])
cov_svm_prec = np.array([0.7852103 , 0.77481568, 0.78118153])
cov_svm_recall = np.array([0.82777455, 0.79874621, 0.82657127])
cov_svm_roc = np.array([0.8693234 , 0.85813217, 0.87016745])
cov_svm_f1 = np.array([0.80593083, 0.78659898, 0.80323569])

adult_svm_val = np.array([0.8452, 0.846 , 0.8444])
adult_svm_acc = np.array([0.84222565, 0.83822286, 0.83926707])
adult_svm_prec = np.array([0.73786789, 0.7174379 , 0.73489175])
adult_svm_recall = np.array([0.5650606 , 0.57151466, 0.55318936])
adult_svm_roc = np.array([0.89260571, 0.88584867, 0.89383485])
adult_svm_f1 = np.array([0.64000454, 0.63621625, 0.63122469])

O_svm_val = np.array([0.9934, 0.9924, 0.9924])
O_svm_acc = np.array([0.99113333, 0.9932    , 0.99326667])
O_svm_prec = np.array([0.91481481, 0.92449355, 0.94357977])
O_svm_recall = np.array([0.85025818, 0.89165187, 0.87073609])
O_svm_roc = np.array([0.99675533, 0.99784277, 0.99772609])
O_svm_f1 = np.array([0.88135593, 0.90777577, 0.90569561])

AM_svm_val = np.array([0.9602, 0.9572, 0.9542])
AM_svm_acc = np.array([0.9624    , 0.96606667, 0.96606667])
AM_svm_prec = np.array([0.96103198, 0.96797297, 0.96727273])
AM_svm_recall = np.array([0.96310261, 0.96341627, 0.96428571])
AM_svm_roc = np.array([0.99276918, 0.99314926, 0.993619  ])
AM_svm_f1 = np.array([0.96206618, 0.96568925, 0.96577691])


# In[3]:


# table 4

KNN_val = (sum(O_knn_val) + sum(cov_knn_val) + sum(adult_knn_val) + sum(AM_knn_val))/12
SVM_val = (sum(O_svm_val) + sum(cov_svm_val) + sum(adult_svm_val) + sum(AM_svm_val))/12
LOGIT_val = (sum(O_log_val) + sum(cov_log_val) + sum(adult_log_val) + sum(AM_log_val))/12


# In[4]:


KNN_val,SVM_val, LOGIT_val


# In[5]:


#Compute Stats for table1

adult_attr = len(adult_hot.columns) - 1 
adult_tr = 5000
adult_tst = adult_hot.shape[0]-5000
adult_percentage = sum(adult_hot.income)/adult_hot.shape[0]

cov_attr = len(cov_bin.columns)-1
cov_tr = 5000
cov_tst = cov_bin.shape[0]-5000
cov_percentage = sum(cov_bin.class2)/cov_bin.shape[0]

lettersO_attr = len(letters_dat.columns)-3
lettersO_tr = 5000
lettersO_tst = letters_dat.shape[0]-5000
lettersO_percentage = sum(letters_dat.O_pos)/letters_dat.shape[0]

lettersAM_attr = len(letters_dat.columns)-3
lettersAM_tr = 5000
lettersAM_tst = letters_dat.shape[0]-5000
lettersAM_percentage = sum(letters_dat.AM_pos)/letters_dat.shape[0]
lettersAM_attr,lettersAM_tr,lettersAM_tst,lettersAM_percentage


# In[6]:


#Compute stats for table2

##knn
knn_acc = (sum(O_knn_acc) + sum(AM_knn_acc) + sum(cov_knn_acc) + sum(adult_knn_acc))/12
knn_prec = (sum(O_knn_prec)+sum(AM_knn_prec)+sum(cov_knn_prec)+sum(adult_knn_prec))/12
knn_recall = (sum(O_knn_recall)+sum(AM_knn_recall) + sum(cov_knn_recall)+sum(adult_knn_recall))/12
knn_roc = (sum(O_knn_roc)+sum(AM_knn_roc)+sum(cov_knn_roc)+sum(adult_knn_roc))/12
knn_f1 = (sum(O_knn_f1) + sum(AM_knn_f1)+ sum(cov_knn_f1)+sum(adult_knn_f1))/12
knn_mean = (knn_acc + knn_prec + knn_recall + knn_roc + knn_f1)/5

##svm 

svm_acc = (sum(O_svm_acc)+sum(AM_svm_acc)+sum(cov_svm_acc)+sum(adult_svm_acc))/12
svm_prec = (sum(O_svm_prec)+sum(AM_svm_prec)+sum(cov_svm_prec)+sum(adult_svm_prec))/12
svm_recall = (sum(O_svm_recall) +sum(AM_svm_recall)+sum(cov_svm_recall)+sum(adult_svm_recall))/12
svm_roc = (sum(O_svm_roc)+sum(AM_svm_roc)+sum(cov_svm_roc)+sum(adult_svm_roc))/12
svm_f1 = (sum(O_svm_f1) + sum(AM_svm_f1)+ sum(cov_svm_f1)+sum(adult_svm_f1))/12
svm_mean = (svm_acc+svm_prec+svm_recall+svm_roc+svm_f1)/5

##logit

log_acc = (sum(O_log_acc)+sum(AM_log_acc)+sum(cov_log_acc)+sum(adult_log_acc))/12
log_prec = (sum(O_log_prec)+sum(AM_log_prec)+sum(cov_log_prec)+sum(adult_log_prec))/12
log_recall = (sum(O_log_recall)+sum(AM_log_recall) + sum(cov_log_recall)+sum(adult_log_recall))/12
log_roc = (sum(O_log_roc)+sum(AM_log_roc)+sum(cov_log_roc)+sum(adult_log_roc))/12
log_f1 = (sum(O_log_f1) + sum(AM_log_f1)+ sum(cov_log_f1)+sum(adult_log_f1))/12
log_mean = (log_acc + log_prec + log_recall + log_roc + knn_f1)/5


# In[7]:


##Compute stats for table3


# In[8]:


KNN_COV = sum((cov_knn_acc + cov_knn_prec + cov_knn_recall + cov_knn_roc + cov_knn_f1)/5)/3
KNN_adult = sum((adult_knn_acc+ adult_knn_prec + adult_knn_recall + adult_knn_roc + adult_knn_f1)/5)/3
KNN_letter1 = sum((O_knn_acc + O_knn_prec + O_knn_recall + O_knn_roc + O_knn_f1)/5)/3
KNN_letter2 = sum((AM_knn_acc + AM_knn_prec + AM_knn_recall+O_knn_recall)/5)/3
KNN_mean = (KNN_COV + KNN_adult + KNN_letter1 + KNN_letter2)/4

SVM_COV = sum((cov_svm_acc + cov_svm_prec + cov_svm_recall + cov_svm_roc + cov_svm_f1)/5)/3
SVM_adult = sum((adult_svm_acc+ adult_svm_prec + adult_svm_recall + adult_svm_roc + adult_svm_f1)/5)/3
SVM_letter1 = sum((O_svm_acc + O_svm_prec + O_svm_recall + O_svm_roc + O_svm_f1)/5)/3
SVM_letter2 = sum((AM_svm_acc + AM_svm_prec + AM_svm_recall+O_svm_recall)/5)/3
SVM_mean = (SVM_COV + SVM_adult + SVM_letter1 + SVM_letter2)/4

LOG_COV = sum((cov_log_acc + cov_log_prec + cov_log_recall + cov_log_roc + cov_log_f1)/5)/3
LOG_adult = sum((adult_log_acc+ adult_log_prec + adult_log_recall + adult_log_roc + adult_log_f1)/5)/3
LOG_letter1 = sum((O_log_acc + O_log_prec + O_log_recall + O_log_roc + O_log_f1)/5)/3
LOG_letter2 = sum((AM_log_acc + AM_log_prec + AM_log_recall+O_log_recall)/5)/3
LOG_mean = (LOG_COV + LOG_adult + LOG_letter1 + LOG_letter2)/4


# In[26]:


#t test for table 3


# In[34]:


svm_cov_ba = (cov_svm_acc + cov_svm_prec + cov_svm_recall + cov_svm_roc + cov_svm_f1)/5
svm_adult_ba = (adult_svm_acc+ adult_svm_prec + adult_svm_recall + adult_svm_roc + adult_svm_f1)/5
svm_letter1 = (O_svm_acc + O_svm_prec + O_svm_recall + O_svm_roc + O_svm_f1)/5
svm_letter2 = (AM_svm_acc + AM_svm_prec + AM_svm_recall+O_svm_recall)/5

knn_cov_ba = (cov_knn_acc + cov_knn_prec + cov_knn_recall + cov_knn_roc + cov_knn_f1)/5
knn_adult_ba = (adult_knn_acc+ adult_knn_prec + adult_knn_recall + adult_knn_roc + adult_knn_f1)/5
knn_letter1_ba = (O_knn_acc + O_knn_prec + O_knn_recall + O_knn_roc + O_knn_f1)/5
knn_letter2_ba = (AM_knn_acc + AM_knn_prec + AM_knn_recall+O_knn_recall)/5

log_cov_ba = (cov_log_acc + cov_log_prec + cov_log_recall + cov_log_roc + cov_log_f1)/5
log_adult_ba = (adult_log_acc+ adult_log_prec + adult_log_recall + adult_log_roc + adult_log_f1)/5
log_letter1_ba = (O_log_acc + O_log_prec + O_log_recall + O_log_roc + O_log_f1)/5
log_letter2_ba = (AM_log_acc + AM_log_prec + AM_log_recall+O_log_recall)/5


# In[48]:


ttest_rel(knn_letter2_ba, svm_letter2)
ttest_rel(svm_letter2,knn_letter2_ba)


# In[40]:


svm_letter1
log_letter1_ba


# In[11]:


##t test for table 2

from scipy.stats import ttest_rel


# In[12]:


knn_acc_vec = np.concatenate((O_knn_acc, AM_knn_acc, cov_knn_acc, adult_knn_acc))
svm_acc_vec = np.concatenate((O_svm_acc, AM_svm_acc, cov_svm_acc, adult_svm_acc)) ##better than knn/logit
logit_acc_vec = np.concatenate((O_log_acc, AM_log_acc, cov_log_acc, adult_log_acc))

knn_prec_vec = np.concatenate((O_knn_prec, AM_knn_prec, cov_knn_prec, adult_knn_prec))
svm_prec_vec = np.concatenate((O_svm_prec, AM_svm_prec, cov_svm_prec, adult_svm_prec)) ##better than knn/logit
logit_prec_vec = np.concatenate((O_log_prec, AM_log_prec, cov_log_prec, adult_log_prec))

knn_recall_vec = np.concatenate((O_knn_recall, AM_knn_recall, cov_knn_recall, adult_knn_recall))
svm_recall_vec = np.concatenate((O_svm_recall, AM_svm_recall, cov_svm_recall, adult_svm_recall)) ##better than knn/logit
logit_recall_vec = np.concatenate((O_log_recall, AM_log_recall, cov_log_recall, adult_log_recall))

knn_roc_vec = np.concatenate((O_knn_roc, AM_knn_roc, cov_knn_roc, adult_knn_roc))
svm_roc_vec = np.concatenate((O_svm_roc, AM_svm_roc, cov_svm_roc, adult_svm_roc)) #better than both
logit_roc_vec = np.concatenate((O_log_roc, AM_log_roc, cov_log_roc, adult_log_roc))

knn_f1_vec = np.concatenate((O_knn_f1,AM_knn_f1,cov_knn_f1,adult_knn_f1))
svm_f1_vec = np.concatenate((O_svm_f1,AM_svm_f1,cov_svm_f1,adult_svm_f1)) #better than both
logit_f1_vec = np.concatenate((O_log_f1,AM_log_f1,cov_log_f1,adult_log_f1))


# In[25]:


## t_tests for table 2


# In[13]:


ttest_rel(svm_acc_vec, logit_acc_vec)


# In[14]:


ttest_rel(svm_acc_vec, knn_acc_vec)


# In[15]:


ttest_rel(svm_prec_vec,knn_prec_vec)


# In[16]:


ttest_rel(svm_prec_vec,logit_prec_vec)


# In[17]:


ttest_rel(svm_recall_vec, knn_recall_vec)


# In[18]:


ttest_rel(svm_recall_vec,logit_recall_vec)


# In[19]:


ttest_rel(svm_roc_vec,knn_roc_vec)


# In[20]:


ttest_rel(svm_roc_vec,logit_roc_vec)


# In[21]:


ttest_rel(svm_f1_vec,knn_f1_vec)


# In[22]:


ttest_rel(svm_f1_vec, logit_f1_vec)


# In[23]:


knn_vec_tot = np.array([.885,.826,.777,.892,.795])
svm_vec_tot = np.array([.899,.850,.804,.936,.824])
log_vec_tot = np.array([.820,.546,.527,.842,.534])


# In[24]:


ttest_rel(svm_vec_tot,knn_vec_tot)


# In[ ]:




