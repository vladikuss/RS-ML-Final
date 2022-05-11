from comet_ml import Experiment
experiment = Experiment(
    api_key="vDQl1ypFBK2U8lhpGHfeRnADG",
    project_name="general",
    workspace="vladikuss",
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from preprocess import *
import Forest as f
import pandas as pd

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=451)
final = []
def low_fit(model):
    model.fit(x_train_sc, y_train) 
    y_pred=model.predict(x_test_sc)

    score=accuracy_score(y_test, y_pred)
    score_train = accuracy_score(y_train, model.predict(x_train_sc))

    auc = model.predict_proba(x_test_sc)
    roc = roc_auc_score(y_test, auc, multi_class='ovr')
    fscore = f1_score(y_test, y_pred, average='macro')

    print(f'Train accuracy: {round(score_train*100,2)}%')
    print(f'Test accuracy: {round(score*100,2)}%')

    metrics = {"Accuracy":score, 'ROC AUC':roc, 'F1':fscore}
    experiment.log_metrics(metrics)

#%%
RF = RandomForestClassifier(n_estimators=295, max_features='auto', max_depth=43, criterion='entropy', n_jobs=-1, random_state=451)
low_fit(RF)
experiment.end()
#%%
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=RF, n_estimators=47, learning_rate=1.448551724137931, algorithm='SAMME.R', random_state=451)
ada = AdaBoostClassifier(base_estimator = RandomForestClassifier())
low_fit(ada)
experiment.end()
#%%
RF.fit(X_sc,y)
testdata = pd.read_csv('data/test.csv', index_col='Id')
testdata = sc(testdata)
testpred = RF.predict(testdata)
testdata['Cover_Type'] = testpred
fin = testdata['Cover_Type']
fin = fin.reset_index()#%%
fin.to_csv('final.csv', index=False)
