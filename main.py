# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from page import streamlitShow 

# %%
df = pd.read_csv("dataset/bankloan.csv")
df = df.drop("ID",axis=1)
df

# %%
X = df.drop('Personal.Loan',axis=1)
y = df['Personal.Loan']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42069)

# %%

lr_default = LogisticRegression(max_iter=400)


lr_penalized1 = LogisticRegression(C=0.3, max_iter=200, penalty='l1', solver='liblinear')
lr_penalized2 = LogisticRegression(C=0.1, max_iter=200 ,penalty='l2', solver='lbfgs')

#Fitting
lr_default.fit(X_train, y_train)
lr_penalized1.fit(X_train, y_train)
lr_penalized2.fit(X_train, y_train)

# %%

rfcla_default = RandomForestClassifier()


rfcla_modified1 = RandomForestClassifier(n_estimators=200 ,max_depth=10, min_samples_split=2, 
                                    min_samples_leaf=1, max_features="log2", random_state=100)
rfcla_modified2 = RandomForestClassifier(n_estimators=100 ,max_depth=10, min_samples_split=2, 
                                    min_samples_leaf=1, max_features="sqrt", random_state=100)

#Fitting
rfcla_default.fit(X_train, y_train)
rfcla_modified1.fit(X_train, y_train)
rfcla_modified2.fit(X_train, y_train)

# %%

lr = LogisticRegression()

param_grid = {
    'C': [0.1, 0.3, 0.5, 1, 5, 10],
    'max_iter': [100, 150, 200, 500],
    'penalty':['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

gridsearch_lr_model = grid_search.best_estimator_

print("Best Parameters for Logistic Regression: ", grid_search.best_params_)
print("Best Cross-Validation Score: {:.2f}".format(grid_search.best_score_))
print("Model score: ", gridsearch_lr_model.score(X_test, y_test))

# %%
rf_classifier = RandomForestClassifier()

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 50, 100],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2],
    'random_state':[1, 42, 100]
}

grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy')

grid_search_rf.fit(X_train, y_train)

gridsearch_rfcla_model = grid_search_rf.best_estimator_

print("Best Parameters for Random Forest Classifier: ", grid_search_rf.best_params_)
print("Best Cross-Validation Score: {:.2f}".format(grid_search_rf.best_score_))
print("Model score: ", gridsearch_rfcla_model.score(X_test, y_test))

# %%
models = [lr_default, lr_penalized1, lr_penalized2, gridsearch_lr_model, rfcla_default, rfcla_modified1, rfcla_modified2, gridsearch_rfcla_model]
model_names = ["Logistic Regression",
               "Logistic Regression (Penalized L1)", 
               "Logistic Regression (Penalized L2)",
               "Logistic Regression (GridSearchCV)",
               "Random Forest Classifier",
               "Random Forest Classifier (Modified features 1)",
               "Random Forest Classifier (Modified features 2)",
               "Random Forest Classifier (GridSearchCV)"]

for model, model_name in zip(models, model_names):
    y_pred = model.predict(X_test)
    print(f"{model_name} score: {model.score(X_test, y_test)}")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    cm=metrics.confusion_matrix(y_test, y_pred,labels=[0, 1])

    df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                    columns = [i for i in ["No","Yes"]])
    plt.figure(figsize = (7,5))
    plt.title(f"{model_name}")
    sns.heatmap(df_cm, annot=True ,fmt='g')
    plt.savefig(f'graphs/{model_name} Confusion Matrix')
    print("\n")

# %%
# ROC Curve and AUC for each model

for model, model_name in zip(models, model_names):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    auc = metrics.auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for  {model_name}')
    plt.legend()
    plt.savefig(f'graphs/ROC Curve for {model_name}')
    plt.show()

streamlitShow(X_train, y_train, lr_penalized2, gridsearch_rfcla_model)


