import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from torch.utils.data import TensorDataset, DataLoader
import torch
import transformers as ppb
import nltk

import Pretraitement
import DataReader



class BERT:

    def read_file(self, file_path = 'offenseval-training-v1.tsv'):
        reader = DataReader.DataReader(file_path)
        data, labels = reader.get_labelled_data()
        return (data, labels)

    def runner(self, batchSize):
        nltk.download("punkt")


        pretrait = Pretraitement.Pretraitement()
        data, labels = tache.read_file()

        df = pd.DataFrame({'data': data, 'labels': labels})

        ## We reduced the size of the dataset sinze our cpu could not handle the full size
        df = df[:1000]


        # For DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        ## For BERT:
        #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        # tokenization
        tokenized = df["data"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
        max_len = max(len(i) for i in tokenized)

        # padding of the tokenized data
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        # attention mask
        attention_mask = np.where(padded != 0, 1, 0)

        # converting to PyTorch tensors
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(df['labels'].values, dtype=torch.long)

        dataset = TensorDataset(input_ids, attention_mask, labels)

        # we create a dataloader in order to separate in batches
        dataloader = DataLoader(dataset, batch_size= batchSize, shuffle=True)

        # batches
        model.eval()  
        features = []
        label_list = []  

        # iterate through each batch to then calculate the model predictions 
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_labels = batch
            
            with torch.no_grad():
                last_hidden_states = model(b_input_ids, attention_mask=b_attention_mask)

            features.append(last_hidden_states[0][:,0,:].cpu().numpy())
            label_list.append(b_labels.cpu().numpy())

        # concatenating all the features and labels back to one data set
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(label_list, axis=0)

        # splittin the data for training and testing
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

        # main model
        lr_clf = LogisticRegression(solver='liblinear', C= 1.0 , max_iter=1000) #C = 5.2, solver='saga', 

        # training the model
        lr_clf.fit(train_features, train_labels)

        print("\nScore with batch size: ", batchSize)
        print(lr_clf.score(test_features, test_labels))
        return lr_clf.score(test_features, test_labels)
        
    
if __name__ == "__main__":
    tache = BERT()
    batch_sizes = [16,32,64,128]
    x = ["Batch size: 16", "Batch size: 32", "Batch size: 64", "Batch size: 128"]
    y = []
    times = []
    
    for i in batch_sizes:
        start = time.time()
        value = tache.runner(i)
        end = time.time()
        y.append(value)
        duration = end - start  
        times.append(duration)


    ## Graph plotting
    fig, ax = plt.subplots()
    bars = ax.bar(x, y)

    ax.set_xlabel('Batch sizes')
    ax.set_ylabel('Score accuracy')
    ax.set_ylim([0,1])
    ax.set_title('BERT accuracy based on batch sizes')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2 - 0.1, yval+0.01, str(yval))

    for i in range(len(batch_sizes)):
        print("Batch size " + str(batch_sizes[i]) + " took " + str(times[i]) + " seconds")

    plt.show()




# 0 - Not offensive
# 1 - Offensive