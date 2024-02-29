import json

for current_dataset in ['HS-Brexit']:  # loop on datasets
    for current_split in ['train']:  # loop on splits, here only train
        current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
        train_data = json.load(open(current_file, 'r', encoding='UTF-8'))
        texts = []  # load data
        for item_id in train_data:  # loop across items for the loaded datasets
            text = train_data[item_id]['text']
            text = text.replace('\t', ' ').replace('\n', ' ').replace('\r',
                                                                      ' ')  # remove tabs and similar from text, so we can have everything on a line
            texts.append(text)
            # print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))
            # labeled_data.append((item_id, text))
for current_dataset in ['HS-Brexit']:  # loop on datasets
    for current_split in ['dev']:  # loop on splits, here only train
        current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
        dev_data = json.load(open(current_file, 'r', encoding='UTF-8'))
        dev_texts = []  # load data
        for item_id in dev_data:  # loop across items for the loaded datasets
            text = dev_data[item_id]['text']
            text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            dev_texts.append(text)  # remove tabs and similar from text, so we can have everything on a line
# print(len(texts))
# print(len(dev_texts))
# print(labeled_data)
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from collections import OrderedDict
from gensim.models import word2vec
from gensim.utils import tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# text=[data[key]['text'] for key in data.keys()]
def soft_solution(data):
    soft_list = []
    for n in data:
        #new_label=data[n]['soft_label']['1']
        label_one = data[n]['soft_label']['1']
        label_zero = data[n]['soft_label']['0']
        new_label = [label_one, label_zero]
        soft_list.append(new_label)
    return soft_list


train_soft_labels = soft_solution(train_data)
dev_labels = soft_solution(dev_data)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, train_soft_labels, test_size=0.2,
                                                                      random_state=None)

# print(f'Index of "happy" is {vocab["happy"]}')
# choose sg=0 or sg=1
# min_count: This is an integer that sets the minimum frequency threshold for words to be included in the vocabulary
# window:maximum distance between the current and predicted word within a sentence. A larger window size means that the model can take more context into account when predicting the target word
tokenized_texts = [list(tokenize(singletext)) for singletext in train_texts]
emb_model = word2vec.Word2Vec(tokenized_texts, sg=1, min_count=1, window=3, vector_size=100, )
# loop through each token and update the frequency count in the dictionary
# happy_embedding = emb_model.wv['Brexit']
# print(emb_model.wv.similar_by_word('Brexit', topn = 5))
# print(emb_model.wv.vectors)
weights = torch.FloatTensor(emb_model.wv.vectors)
# vocab = emb_model.wv.index_to_key


# print(weights)
# print(weights.shape)
# embedding_vector = emb_model.wv.get_vector("Brexit")
# print(embedding_vector)
# snippet a:neural network
# Convert tokenized text to list of indices
#print(emb_model.wv.key_to_index['Brexit'])


# def encode_text(tweet):
#   tokens = tokenize(tweet)  # Tokenize one document


#    input_ids = []
#   for token in tokens:
#      if str.lower(token) in vocab:  # Skip words from the dev/test set that are not in the vocabulary.
# input_ids.append(vocab[str.lower(token)] + 1)  # +1 is needed because we reserve 0 as a special character

# list=[]


# return input_ids
# define一下使得train，dev，test都能用它
def encode_text(raw_text, emb_model):
    tokenized_texts = [list(tokenize(single_text)) for single_text in raw_text]
    emb_model = word2vec.Word2Vec(tokenized_texts, sg=1, min_count=1, window=3, vector_size=100, )
    listOfList = []
    for tweet in tokenized_texts:
        listOfIndices = []
        for tokens in tweet:
            listOfIndices.append(emb_model.wv.key_to_index[tokens])
        listOfList.append(listOfIndices)
    return listOfList


train_ids = encode_text(train_texts, emb_model)

print("length of train_id:", len(train_ids))
rv_l = [len(doc) for doc in train_ids]
# print('Mean of the document length: {}'.format(np.mean(rv_l)))
# print('Median of the document length: {}'.format(np.median(rv_l)))

# print('Maximum document length: {}'.format(np.max(rv_l)))
# plt.hist(rv_l)
# b:make all of our documents have the same number of tokens and padding the code
sequence_length = 20  # truncate all docs longer than this. Pad all docs shorter than this.


# don't know how to choose sequence length?
def pad_text(input_ids, sequence_length):
    if len(input_ids) >= sequence_length:
        input_ids = input_ids[:sequence_length]
    else:
        input_ids = [0] * (sequence_length - len(input_ids)) + input_ids
    ##########
    return np.array(input_ids)


# The will call pad_text for every document in the dataset
train_ids = [pad_text(input_ids, sequence_length) for input_ids in train_ids]
# print("trainid after padding:",len(train_ids))
# convert from the Huggingface format to a TensorDataset so the mini-batch sampling functionality
batch_size = 100


def convert_to_data_loader(ids, labels):
    # convert from list to tensor
    input_tensor = torch.from_numpy(np.array(ids))
    label_tensor = torch.from_numpy(np.array(labels)).long()
    tensor_dataset = TensorDataset(input_tensor, label_tensor)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    return loader


# print(train_labels)，num class没用
#num_classes = 7  # number of possible labels in the sentiment analysis task
# 这里定义一个data——loader使得这段代码简洁一些
train_loader = convert_to_data_loader(train_ids, train_labels)
dev_ids = encode_text(dev_texts, emb_model)
dev_ids = [pad_text(input_ids, sequence_length) for input_ids in dev_ids]
dev_loader = convert_to_data_loader(dev_ids, dev_labels)
test_ids = encode_text(test_texts, emb_model)
test_ids = [pad_text(input_ids, sequence_length) for input_ids in test_ids]
test_loader = convert_to_data_loader(test_ids, test_labels)
# embedding_size = emb_model.vector_size
# sequence_length = max(len(text) for text in tokenized_texts)

# embedding = nn.Embedding.from_pretrained(weights)
# snippetb:foward method
vocab_size = 3131
embedding_size = emb_model.vector_size  # number of dimensions for embeddings
hidden_size = 100


class LSTMTextClassifier(nn.Module):

    def __init__(self, vocab_size, sequence_length, embedding_size, hidden_size):
        super(LSTMTextClassifier, self).__init__()

        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        # self.embedding.weight = nn.Parameter(weights)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.embedding_layer.weight = nn.Parameter(weights, requires_grad=False)
        # self.embedding.weight = nn.Parameter(weights)
        self.lstm_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True,bidirectional=False )  # Hidden layer

        self.output_layer=nn.Linear(hidden_size, 1)
        ##########

    def forward(self, input_words):
        # Input dimensions are:  (batch_size, seq_length)
        embedded_words = self.embedding_layer(input_words)  # (batch_size, seq_length, embedding_size)

        # flatten the sequence of embedding vectors for each document into a single vector.
        #embedded_words = embedded_words.reshape(embedded_words.shape[0],
                                                #self.sequence_length * self.embedding_size)  # batch_size, seq_length*embedding_size
        #print(embedded_words.shape)

        # Apply LSTM to the embedded sequence

        lstm_output, (h_n, c_n) = self.lstm_layer(embedded_words)
        #lstm_output=
        # Use the last hidden state as input to the output layer
        logits = self.output_layer(lstm_output[:, -1])
        output=torch.sigmoid(logits)

        return output


ff_classifier_model = LSTMTextClassifier(vocab_size, sequence_length, embedding_size, hidden_size)
#ff_classifier_model.forward()
from torch import optim


def cross_entropy(targets_soft, predictions_soft, epsilon=1e-12):
    # 把prediction的值限制在epsilon 和1. -epsilion 之间
    predictions = torch.clamp(predictions_soft, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    #q=torch.log(predictions + 1e-9)
    #ce = -torch.sum(targets_soft * q+(1-targets_soft)*(1-q)) / N
    #ce = -torch.sum(targets_soft * torch.log(predictions + 1e-9) + (1 - targets_soft) * (1 - torch.log(predictions + 1e-9))) / N
    ce = -torch.sum(targets_soft * torch.log(predictions + 1e-9)) / N
    return ce


# targets_soft=[[0.6,0.4],[0.5,0.5]]
# prediction_soft=[[0.7,0.3],[0.3,0.7]]

# a=cross_entropy(targets_soft,prediction_soft)


# soft_list=targets_soft(train_data)
# batch_loss=cross_entropy(soft_list,output)
num_epochs = 20


def train_nn(num_epochs, model, train_dataloader, dev_dataloader):
    learning_rate = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dev_losses = []
    for e in range(num_epochs):
        train_losses = []
        model.train()
        for i, (batch_input_ids, batch_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(batch_input_ids)
            batch_loss = cross_entropy(batch_labels, output, 1e-12)
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.item())
        print("Epoch: {}/{}".format((e + 1), num_epochs),
              "Training Loss: {:.4f}".format(np.mean(train_losses)))
        model.eval()
        dev_losses_epoch = []
        all_dev_output=[]
        for dev_input_ids, dev_labels in dev_dataloader:
            dev_output = model(dev_input_ids)
            all_dev_output.append(dev_output)
            dev_loss = cross_entropy(dev_labels, dev_output)
            dev_losses_epoch.append(dev_loss.item())
        dev_losses.append(np.mean(dev_losses_epoch))
        print("Epoch: {}/{}".format((e + 1), num_epochs),
              "Validation Loss: {:.4f}".format(dev_losses[-1]))
        all_dev_output= torch.cat(all_dev_output, dim=0)
    return model, dev_losses, all_dev_output


trained_model, dev_losses, output = train_nn(num_epochs, ff_classifier_model, train_loader, dev_loader)

plt.title("softlabel")
plt.plot(dev_losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss using  lstm')
plt.show()
