
# coding: utf-8

# In[38]:

import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#check whether gpu is available
use_gpu=torch.cuda.is_available()

# read file
f = open('input.txt')
result = list()
result.append(f.read())
#for line in f.readlines():    
    #print(line)
    #line = line.strip()                
 #   result.append(line)              
#print(len(result))
#indices_start=[i for i in range(len(result)) if result[i].strip()=='<start>']
#num_tune=len(indices_start)
#tune=list()

#for i in range(len(result)-1):
#    tune.append(''.join(result[indices_start[i]:indices_start[i+1]]))
#tune.append(''.join(result[indices_start[-1]:-1]))

training_data = []
training_targets = []
test_data = []
test_targets = []
sentencelen = 25
for i in range(len(result)):
    #leave one more for validation set
    num_character=len(result[i])
    #k = num_character%sentencelen
    k = num_character//sentencelen
    for j in range(k):
        training_data.append(result[i][j*sentencelen:(j+1)*sentencelen])
        piece = result[i][j*sentencelen+1:(j+1)*sentencelen+1]
        if(len(piece)<sentencelen):
            piece = piece+" "*(sentencelen-len(piece))
        training_targets.append(piece)
    end = ""
    end1 = ""
    if(k*sentencelen<num_character):
        end = result[i][k*sentencelen:len(result[i])]
        if(k*sentencelen+1<num_character):
            end1 = result[i][k*sentencelen+1:len(result[i])]
            end1 = end1+ ' ' * (sentencelen - len(end1))
        else:
            end1 = ' '* sentencelen
        end = end + ' '* (sentencelen - len(end))
        end1 = end1+ ' ' * (sentencelen - len(end1))
        training_data.append(end)
        training_targets.append(end1)
        #print('0'* sentencelen)
k = int(len(training_data)*0.8)
test_data=training_data[k:]
test_targets=training_targets[k:]
training_data=training_data[:k]
training_targets=training_targets[:k]
#word to index
word_to_ix = {}
ix_to_word = {}
for sent in training_data:
    for characters in sent:
        if characters not in word_to_ix:
            word_to_ix[characters] = len(word_to_ix) 
            ix_to_word[len(word_to_ix)-1] = characters


# In[39]:


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    idxs = torch.LongTensor(idxs)
    #print(len(tensor))
    #tensor = tensor.view(batch_size,len(tensor)/batch_size)
    return autograd.Variable(idxs)
def one_hot(ids, depth):
    """
    ids: Variable
    out_tensor:FloatTensor shape:[sentencelen, depth]
    """
    out_tensor=torch.Tensor(len(ids),depth)
    #if not isinstance(ids, (list, np.ndarray)):
     #   raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids.data.cpu().numpy()).view(-1,1)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids,value=1. )
    if torch.cuda.is_available():
        out_tensor=out_tensor.cuda()
    return autograd.Variable(out_tensor)
    


# In[40]:


EMBEDDING_DIM = len(word_to_ix)
HIDDEN_DIM = 100

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if use_gpu:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),\
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),\
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence , hidden):
        #print(sentence)
        #embeds = self.word_embeddings(sentence)
        embeds=one_hot(sentence,EMBEDDING_DIM)
        lstm_out, hidden = self.lstm(
            embeds.view(len(sentence),1,-1), hidden)
        #print(self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #tag_scores = F.softmax(tag_space)
        return tag_space,hidden
    


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
# model.load_state_dict(torch.load('mytraining.pth'))
loss_function = nn.CrossEntropyLoss()
if use_gpu:
    model.cuda()
    loss_function.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)
#inputs = prepare_sequence(training_data, word_to_ix)
error_train = []
error_valid = []
loss_f = 0
k = 0
epoch = 0


# In[41]:


def evaluate(prime_str="<start>", predict_len=10000, temperature=1):
    hidden = model.init_hidden()
    prime_input = prepare_sequence(prime_str, word_to_ix)
    predicted = prime_str
    model(prime_input,hidden)
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]
    for p in range(predict_len):
        model.lstm.flatten_parameters()
        output, hidden = model(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = ix_to_word[top_i]
        predicted += predicted_char
        inp = prepare_sequence(predicted_char, word_to_ix)
        end = predicted[len(predicted)-len("<end>"):]
        #if(end == "<end>"):
         #   break
    print(predicted)
    return predicted
# prime_input = prepare_sequence("<start>", word_to_ix)
# print(prime_input)
# res = evaluate()

