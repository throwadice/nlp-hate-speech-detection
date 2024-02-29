from annotators import Cluster0_data,Cluster1_data,Cluster2_data,Cluster3_data,Cluster4_data,Cluster5_data
import fullconnectionmodel
import json
import lstmmodel
def load_and_filter_train_data(dataset,split,cluster_data):
    for current_dataset in dataset:                         # loop on datasets
      for current_split in split:                                                                 # loop on splits, here only train
        current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
        train_data = json.load(open(current_file,'r', encoding = 'UTF-8'))
        new_train_data = {}
        if cluster_data is None:  # 如果cluster_data参数为空，直接保存所有train_data到new_train_data
            new_train_data = train_data
        else:
         #new_train_data = {}
         for key, value in train_data.items():
            # 如果当前的key值包含在cluster0_list中，则将该项保存到新字典中
            #for string in cluster_data:
                if key in cluster_data:
                 new_train_data[key] = value

        # 保留部分字典，使用新字典替换原来的train_data
        #train_data = new_train_data
        texts = []# load data
        for item_id in new_train_data:                                                                          # loop across items for the loaded datasets
          text = new_train_data[item_id]['text']
          text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')   # remove tabs and similar from text, so we can have everything on a line
          texts.append(text)
          #print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))
          #labeled_data.append((item_id, text))
    return new_train_data,texts
def dev_loader(dataset,train_data):
    for current_dataset in dataset:  # loop on datasets
        for current_split in ['dev']:  # loop on splits, here only train
            current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
            dev_data = json.load(open(current_file, 'r', encoding='UTF-8'))
            #dev_length = len(train_data) // 4
            new_dict={}
            if train_data is None:  # 如果train_data参数为空，直接保存所有dev_data
                new_dict=dev_data
            else:
             dev_length = len(train_data) // 4
             for i, key in enumerate(dev_data.keys()):
                if i >= dev_length:
                    break
                new_dict[key] = dev_data[key]
            #dev_data = new_dict
            dev_texts = []  # load data
            for item_id in new_dict:  # loop across items for the loaded datasets
              text = new_dict[item_id]['text']
              text = text.replace('\t', ' ').replace('\n', ' ').replace('\r',' ')
              dev_texts.append(text)
    return new_dict,dev_texts
dev_data,dev_texts=dev_loader(['HS-Brexit'],train_data=None)
train_data0,texts0=load_and_filter_train_data(['MD-Agreement'], ['train'], Cluster0_data)
dev_data0,dev_texts0=dev_loader(['MD-Agreement'],train_data0)
dev_ouput0=lstmmodel.runlstmmodel(train_data0,dev_data,texts0,dev_texts,cluster_num=0,hidden_size=70,num_epochs=10)
#fullconnectionmodel.run_model(train_data0,dev_data0,texts0,dev_texts0,cluster_num=0,hidden_size=70,num_epochs=10)
#train_data,texts=load_and_filter_train_data(['HS-Brexit'],['train'])
#dev_data,dev_texts=dev_loader(['HS-Brexit'],train_data=None)
#fullconnectionmodel.run_model(train_data, dev_data, texts, dev_texts)

train_data1,texts1=load_and_filter_train_data(['MD-Agreement'], ['train'], Cluster1_data)
dev_data1,dev_texts1=dev_loader(['MD-Agreement'],train_data0)
dev_ouput1=lstmmodel.runlstmmodel(train_data1, dev_data, texts1, dev_texts,cluster_num=1,hidden_size=80,num_epochs=13)
#fullconnectionmodel.run_model(train_data1, dev_data1, texts1, dev_texts1,cluster_num=1,hidden_size=80,num_epochs=13)
#lstm_model=lstm.LSTMTextClassifier
train_data2,texts2=load_and_filter_train_data(['MD-Agreement'], ['train'], Cluster1_data)
dev_data2,dev_texts2=dev_loader(['MD-Agreement'],train_data2)
dev_ouput2=lstmmodel.runlstmmodel(train_data2, dev_data, texts2, dev_texts,cluster_num=2,hidden_size=150,num_epochs=20)
