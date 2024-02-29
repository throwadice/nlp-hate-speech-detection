
import torch
#===== snippet 1: how to read data
# with these few lines you can loop across all datasets and splits and load data 

import json
import pandas as pd
for current_dataset in [ 'HS-Brexit'] :                        # loop on datasets
  for current_split in ['train','dev']:                                                           # loop on splits
    current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json' 
    data = json.load(open(current_file,'r', encoding = 'UTF-8'))


#===== snippet 2: how to read data and save text, soft evaluation and hard evaluation in a different file for each dataset/split
# with these few lines you can loop across all datasets and splits (here only the train) and
# extract (and print) the info you need 
# here we print: dataset,split,id,lang,hard_label,soft_label_0,soft_label_1,text in a tab separated format
# note: each item_id in the dataset for each split is numbered starting from "1"

import json

print("Dataset\tSplit\tId\tLang\tHard_label\tSoft_label_0\tSoft_label_1\tText")                   # print header

for current_dataset in [ 'HS-Brexit']:                         # loop on datasets
  for current_split in ['train']:                                                                 # loop on splits, here only train 
    current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file 
    data = json.load(open(current_file,'r', encoding = 'UTF-8'))                                  # load data 
    for item_id in data:                                                                          # loop across items for the loaded datasets                                                                                                  
      text = data[item_id]['text']        
      text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')   # remove tabs and similar from text, so we can have everything on a line
      #print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))


#对train set, test set进行train test split，并fit其中hard label
#怎么fit{}
#str(data[item_id]['soft_label']["1"])
#模型建立完成后，用test set 进行测试 定义predition是个什么变量
#===== snippet 3: ConvAbuse dataset text from string to conversation
# only for the ConvAbuse dataset, the field "text" is a conversation that for representational purposes has been "stringified". To put it back to it's conversation form you can run
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

text=[data[key]['text'] for key in data.keys()]
hard_labels = [data[key]['hard_label'] for key in data.keys()]
hard_labels = np.array(hard_labels).astype(int)
#print(type(hard_labels))
docs_train, docs_test, y_train, y_test = train_test_split(text, hard_labels, test_size=0.2,random_state=None)
count_vect = CountVectorizer()
tfidf_transformer = TfidfVectorizer()
linear=LinearSVC()


clf = Pipeline([
        ('vect', tfidf_transformer),
        ('clf', LinearSVC(C=1000)),
    ])
clf.fit(docs_train,y_train)
#  Predict the outcome on the testing set in a variable named y_predicted
y_predicted = clf.predict(docs_test)
# Print the classification report
#print(metrics.classification_report(y_test, y_predicted))
#print(clf.get_params().keys())
parameters = {
    'clf__C': [1000,500,800,.1500,3000],
    'clf__penalty': ['l2'],
    'clf__loss': ['hinge', 'squared_hinge'],
    'clf__dual': [True],
    'clf__max_iter': [500, 1000, 1500],
    'clf__tol': [1e-3, 1e-4, 1e-5]
}
grid_search = GridSearchCV(clf, parameters, scoring='accuracy')
grid_search.fit(docs_train, y_train)
y_pred_acc = grid_search.predict(docs_test)
#print(metrics.classification_report(y_test, y_pred_acc))

#snippet 3:
#X_train_counts.shape
#current_dataset = 'HS-Brexit'
#for current_split in ['train', 'dev']:                                                            # loop on splits, here only train
    #current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
    #f_out = open('./'+d+'_'+k+'_conversation.json','w')
    #data = json.load(open(current_file,'r', encoding = 'UTF-8'))                                  # load data
    #for item_id in data:                                                                          # loop across items for the loaded datasets
        #data[item_id]['text'] = json.dumps(data[item_id]['text'])
    #f_out.write(json.dumps(data, indent = 4))
    #f_out.close()


# snippet 4: function that calculates cross-entropy



def cross_entropy(targets_soft, predictions_soft, epsilon = 1e-12):
    predictions = np.clip(predictions_soft, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets_soft*np.log(predictions+1e-9))/N
    return ce

#print(f"Test entropy: {test_entropy:.4f}")
#plot the orginal distribution
from matplotlib import pyplot

#plot the predicted distribution

#test my model given a raw text
#

#snippet 5: function that calculates weighted averaged f1


def f1_metric(targets_hard, prediction_hard):
    f1_wa = metrics.f1_score(targets_hard, prediction_hard, average = 'micro')
    return f1_wa



#===== snippet 6: how to call the scoring functions. 
# Where 'mytruthfile' is a file, with the same format of the ArMIS_results.tsv, containing the true labels. 


#def get_data (myfile):
    #soft = list()
    #hard = list()
    #with open(myfile,'r') as f:
     #       for line in f:

     #           line=line.replace('\n','')
    #
      #          parts=line.split('\t')
     #           soft.append([float(parts[1]),float(parts[2])])

       #         hard.append(parts[0])
    #return(soft,hard)


#soft_ref, hard_ref = get_data(mytruthfile)

#soft_pred, hard_pred = get_data('./majority_baseline_practicephase/res/ArMIS_results.tsv')            # example of a result file
#soft_score = cross_entropy(soft_ref,soft_pred)
hard_score = cross_entropy(y_test,y_predicted )
f1_score=f1_metric(y_test,y_predicted)
print('cross entropy for hard labels:',hard_score)
print('f1_metric for hard labels:',f1_metric)