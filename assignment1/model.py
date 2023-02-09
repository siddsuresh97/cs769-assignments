import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    # load pre-trained word embeddings
    emb = np.zeros((len(vocab), emb_size))
    with zipfile.ZipFile(emb_file) as zf:
        with zf.open(emb_file.split('/')[-1].replace('.zip', '')) as f:
            for line in f:
                line = line.decode('utf-8').split()
                word = line[0]
                if word in vocab:
                    emb[vocab[word]] = np.array(line[1:], dtype=np.float32)
    return emb
    # raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.output_dim = self.tag_size
        self.hidden_dim = self.args.hid_size
        self.layers = nn.ModuleList()
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        current_dim = self.args.emb_size
        for layer in range(self.args.hid_layer):
            self.layers.append(nn.Linear(current_dim, self.args.hid_size))
            current_dim = self.args.hid_size
        self.layers.append(nn.Linear(current_dim, self.output_dim))

        # raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        nn.init.xavier_normal_(self.embedding.weight)
        for layer in self.layers:
            #do xavier normal
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        # raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # load pre-trained word embeddings
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        # copy the embeddings to the embedding layer
        self.embedding.weight.data.copy_(torch.from_numpy(emb))
        # set the embedding layer to be untrainable
        self.embedding.weight.requires_grad = False
        # raise NotImplementedError()

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """ 
        # change type to float
        # if self.args.word_drop > 0:
        #     mask = torch.rand(*x.size()).to(x.device) > self.args.word_drop
        #     x = x.where(mask, torch.zeros_like(x))
        x = self.embedding(x)
        # add embedding dropout
        # x = nn.Dropout(p=self.args.emb_drop)(x)
        if self.args.pooling_method == 'avg':
            x = torch.mean(x, dim=1)
        elif self.args.pooling_method == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.args.pooling_method == 'sum':
            x = torch.sum(x, dim=1)
        else:
            raise ValueError('Invalid pooling method')
        for layer in self.layers[:-1]:
            x = layer(x)
            # add dropout
            # x = nn.Dropout(p=self.args.hid_drop)(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x
        # raise NotImplementedError()
