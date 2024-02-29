import json

for current_dataset in [ 'HS-Brexit']:                         # loop on datasets
  for current_split in ['train']:                                                                 # loop on splits, here only train
    current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
    train_data = json.load(open(current_file,'r', encoding = 'UTF-8'))
    texts = []# load data
    for item_id in train_data:                                                                          # loop across items for the loaded datasets
      text = train_data[item_id]['text']
      text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')   # remove tabs and similar from text, so we can have everything on a line
      texts.append(text)
      #print('\t'.join([current_dataset, current_split, item_id, data[item_id]['lang'], str(data[item_id]['hard_label']), str(data[item_id]['soft_label']["0"]), str(data[item_id]['soft_label']["1"]), text]))
      #labeled_data.append((item_id, text))
for current_dataset in ['HS-Brexit']:  # loop on datasets
    for current_split in ['dev']:  # loop on splits, here only train
        current_file = './' + current_dataset + '_dataset/' + current_dataset + '_' + current_split + '.json'  # current file
        dev_data = json.load(open(current_file, 'r', encoding='UTF-8'))
        dev_texts = []  # load data
        for item_id in dev_data:  # loop across items for the loaded datasets
          text = dev_data[item_id]['text']
          text = text.replace('\t', ' ').replace('\n', ' ').replace('\r',' ')
          dev_texts.append(text)                                                                   # remove tabs and similar from text, so we can have everything on a line
#print(len(texts))
#print(len(dev_texts))
#print(labeled_data)
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
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
#text=[data[key]['text'] for key in data.keys()]
def soft_solution(data):
  soft_list=[]
  for n in data:
    #new_label=0.5*data[n]['soft_label']['0']+0.5*data[n]['soft_label']['1']
    new_label =  data[n]['soft_label']['1']
    soft_list.append(new_label)
  return soft_list
train_soft_labels=soft_solution(train_data)
dev_labels=soft_solution(dev_data)
train_texts, test_texts, train_labels, test_labels=train_test_split(texts, train_soft_labels, test_size=0.2,random_state=None)

#print(f'Index of "happy" is {vocab["happy"]}')
#choose sg=0 or sg=1
#min_count: This is an integer that sets the minimum frequency threshold for words to be included in the vocabulary
#window:maximum distance between the current and predicted word within a sentence. A larger window size means that the model can take more context into account when predicting the target word
tokenized_texts = [list(tokenize(singletext)) for singletext in train_texts]
emb_model = word2vec.Word2Vec(tokenized_texts, sg=1, min_count=1, window=3, vector_size=100,)
        # loop through each token and update the frequency count in the dictionary
#happy_embedding = emb_model.wv['Brexit']
#print(emb_model.wv.similar_by_word('Brexit', topn = 5))
#print(emb_model.wv.vectors)
weights = torch.FloatTensor(emb_model.wv.vectors)
#vocab = emb_model.wv.index_to_key






#print(weights)
#print(weights.shape)
#embedding_vector = emb_model.wv.get_vector("Brexit")
#print(embedding_vector)
#snippet a:neural network
# Convert tokenized text to list of indices
print(emb_model.wv.key_to_index['Brexit'])


#def encode_text(tweet):
 #   tokens = tokenize(tweet)  # Tokenize one document


#    input_ids = []
 #   for token in tokens:
  #      if str.lower(token) in vocab:  # Skip words from the dev/test set that are not in the vocabulary.
           # input_ids.append(vocab[str.lower(token)] + 1)  # +1 is needed because we reserve 0 as a special character

#list=[]


    #return input_ids
#define一下使得train，dev，test都能用它
def encode_text(raw_text,emb_model):
    tokenized_texts = [list(tokenize(single_text)) for single_text in raw_text]
    emb_model=word2vec.Word2Vec(tokenized_texts, sg=1, min_count=1, window=3, vector_size=100,)
    listOfList = []
    for tweet in tokenized_texts:
        listOfIndices = []
        for tokens in tweet:
            listOfIndices .append(emb_model.wv.key_to_index[tokens])
        listOfList.append(listOfIndices)
    return listOfList
train_ids = encode_text(train_texts,emb_model)



print("length of train_id:",len(train_ids))
rv_l = [len(doc) for doc in train_ids]
print('Mean of the document length: {}'.format(np.mean(rv_l)))
print('Median of the document length: {}'.format(np.median(rv_l)))
print('Maximum document length: {}'.format(np.max(rv_l)))
#plt.hist(rv_l)
#b:make all of our documents have the same number of tokens and padding the code
sequence_length = 20  # truncate all docs longer than this. Pad all docs shorter than this.
#don't know how to choose sequence length?
def pad_text(input_ids,sequence_length):


    if len(input_ids) >= sequence_length:
        input_ids = input_ids[:sequence_length]
    else:
        input_ids = [0]*(sequence_length-len(input_ids)) + input_ids
    ##########
    return np.array(input_ids)

# The will call pad_text for every document in the dataset
train_ids = [pad_text(input_ids,sequence_length) for input_ids in train_ids]
#print("trainid after padding:",len(train_ids))
# convert from the Huggingface format to a TensorDataset so the mini-batch sampling functionality
batch_size = 100
def convert_to_data_loader(ids, labels, num_classes):
    # convert from list to tensor
    input_tensor = torch.from_numpy(np.array(ids))
    label_tensor = torch.from_numpy(np.array(labels)).long()
    tensor_dataset = TensorDataset(input_tensor, label_tensor)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    return loader

#print(train_labels)
num_classes = len(np.unique(train_labels))   # number of possible labels in the sentiment analysis task
#这里定义一个data——loader使得这段代码简洁一些
train_loader = convert_to_data_loader(train_ids, train_labels, num_classes)
dev_ids=encode_text(dev_texts,emb_model)
dev_ids = [pad_text(input_ids,sequence_length) for input_ids in dev_ids ]
dev_loader = convert_to_data_loader(dev_ids, dev_labels, num_classes)
test_ids=encode_text(test_texts,emb_model)
test_ids = [pad_text(input_ids,sequence_length) for input_ids in test_ids ]
test_loader = convert_to_data_loader(test_ids, test_labels, num_classes)
#embedding_size = emb_model.vector_size
#sequence_length = max(len(text) for text in tokenized_texts)

#embedding = nn.Embedding.from_pretrained(weights)
#snippetb:foward method
vocab_size = 3131
embedding_size = emb_model.vector_size  # number of dimensions for embeddings
hidden_size = 15
class FFTextClassifier(nn.Module):

    def __init__(self, vocab_size,sequence_length, embedding_size, hidden_size, num_classes):
        super(FFTextClassifier, self).__init__()

        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        #self.embedding.weight = nn.Parameter(weights)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.embedding_layer.weight = nn.Parameter(weights)
        #self.embedding.weight = nn.Parameter(weights)
        self.hidden_layer = nn.Linear(embedding_size * sequence_length, hidden_size)  # Hidden layer
        self.activation = nn.ReLU()  # Hidden layer
        #self.output_layer = nn.Linear(hidden_size, num_classes)  # Full connection layer
        self.output_layer=nn.Softmax()
        ##########

    def forward(self, input_words):
        # Input dimensions are:  (batch_size, seq_length)
        embedded_words = self.embedding_layer(input_words)  # (batch_size, seq_length, embedding_size)

        # flatten the sequence of embedding vectors for each document into a single vector.
        embedded_words = embedded_words.reshape(embedded_words.shape[0],
                                                self.sequence_length * self.embedding_size)  # batch_size, seq_length*embedding_size

        ### ADD THE MISSING LINES HERE
        z = self.hidden_layer(embedded_words)  # (batch_size, seq_length, hidden_size)
        h = self.activation(z)  # (batch_size, seq_length, hidden_size)
        output = self.output_layer(h)  # (batch_size, num_classes)

        # Notice we haven't applied a softmax activation to the output layer -- it's not required by Pytorch's loss function.

        return output
ff_classifier_model = FFTextClassifier(vocab_size,sequence_length, embedding_size, hidden_size, num_classes)
from torch import optim

#snippet 4: function that calculates cross-entropy
def cross_entropy(targets_soft, predictions_soft, epsilon = 1e-12):
    predictions = np.clip(predictions_soft, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets_soft*np.log(predictions+1e-9))/N
    return ce
def train_nn(num_epochs, model, train_dataloader, dev_dataloader):
    learning_rate = 0.0005  # learning rate for the gradient descent optimizer, related to the step size

    loss_fn = nn.CrossEntropyLoss()  # create loss function object
    #loss_fn= cross_entropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # create the optimizer

    dev_losses = []

    for e in range(num_epochs):
        # Track performance on the training set as we are learning...
        total_correct = 0
        total_trained = 0
        train_losses = []

        model.train()  # Put the model in training mode.

        for i, (batch_input_ids, batch_labels) in enumerate(train_dataloader):
            # Iterate over each batch of data
            # print(f'batch no. = {i}')

            optimizer.zero_grad()  # Reset the optimizer

            # Use the model to perform forward inference on the input data.
            # This will run the forward() function.
            output = model(batch_input_ids)

            # Compute the loss for the current batch of data
            batch_loss = loss_fn(output, batch_labels)

            # Perform back propagation to compute the gradients with respect to each weight
            batch_loss.backward()

            # Update the weights using the compute gradients
            optimizer.step()

            # Record the loss from this sample to keep track of progress.
            train_losses.append(batch_loss.item())

            # Count correct labels so we can compute accuracy on the training set
            #argmax把最大的设置为1，其他为0
            #predicted_labels = output.argmax(1)
            predicted_labels=output.Softmax()
            total_correct += (predicted_labels == batch_labels).sum().item()
            total_trained += batch_labels.size(0)

        train_accuracy = total_correct / total_trained * 100

        print("Epoch: {}/{}".format((e + 1), num_epochs),
              "Training Loss: {:.4f}".format(np.mean(train_losses)),
              "Training Accuracy: {:.4f}%".format(train_accuracy))

        model.eval()  # Switch model to evaluation mode
        total_correct = 0
        total_trained = 0

        dev_losses_epoch = []

        for dev_input_ids, dev_labels in dev_dataloader:
            dev_output = model(dev_input_ids)
            dev_loss = loss_fn(dev_output, dev_labels)



            # Save the loss on the dev set
            dev_losses_epoch.append(dev_loss.item())
            ####

            # Count the number of correct predictions
            predicted_labels = dev_output.argmax(1)
            total_correct += (predicted_labels == dev_labels).sum().item()
            total_trained += dev_labels.size(0)

        ### Add your own code here

        dev_losses.append(np.mean(dev_losses_epoch))

        ###

        print("Epoch: {}/{}".format((e + 1), num_epochs),
              "Validation Loss: {:.4f}".format(dev_losses[-1]))

    return model, dev_losses

num_epochs = 20
trained_model, dev_losses = train_nn(num_epochs, ff_classifier_model, train_loader, dev_loader)

import matplotlib.pyplot as plt

plt.plot(dev_losses)
plt.xlabel('epochs')
plt.ylabel('loss')


plt.show()

