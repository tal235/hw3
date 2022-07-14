import gensim
import pandas as pd
import torch
from torch.utils.data import Dataset

from consts import *


class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path
        # Load data to dataframe
        self.df = pd.read_csv(file_path)

        # Split to sentences and labels
        self.text = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

    # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            tokenized_text = self.df[TEXT].apply(lambda text: gensim.utils.tokenize(text, lowercase=True))
            # Create a set of all the unique tokens in the text
            self.vocab = set([word for sentence in tokenized_text.to_list() for word in sentence])

        else:
            self.vocab = vocab

        # Add the UNK token to the vocab
        self.vocab.add(UNK_TOKEN)
        self.vocab.add(PAD_TOKEN)
        # Set the vocab size
        self.vocab_size = len(self.vocab)

        # Create a dictionary mapping tokens to indices
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for i, token in enumerate(self.vocab)}

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] = self.df.apply(lambda row: self.tokenize(row), axis=1)


    def __len__(self):
        # Return the length of the dataset
        return len(self.df)


    def __getitem__(self, idx):
        # Get the row at idx
        label = self.labels[idx]
        input_ids = self.df[INPUT_IDS]
        # return the input_ids and the label as tensors, make sure to convert the label type to a long
        return torch.tensor(input_ids), torch.tensor(label, dtype=torch.long)


    def tokenize(self, text):  # I think it is finished (TAL)
        input_ids = []
        # Tokenize the text using gensim.utils.tokenize(text, lowercase=True)
        for i, token in enumerate(gensim.utils.tokenize(text[TEXT], lowercase=True)):
            # Make sure to trim sequences to max_seq_length
            if i >= self.data_args.max_seq_length:
                break
            # Gets the token id, if unknown returns self.unk_token

            if token in self.token2id:
                input_ids.append(self.token2id[token])
            else:
                input_ids.append(self.token2id[UNK_TOKEN])

                # Pad
        for i in range(self.data_args.max_seq_length - len(input_ids)):
            input_ids.append(self.token2id[PAD_TOKEN])

        return input_ids# .LongTensor(input_ids)
