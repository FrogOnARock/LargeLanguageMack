import tiktoken
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

def _get_data():
       url = ("https://raw.githubusercontent.com/rasbt/"
              "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
              "the-verdict.txt")
       file_path = "the-verdict.txt"
       urllib.request.urlretrieve(url, file_path)

class GPTDatasetV1(Dataset):
       def __init__(self, txt, tokenizer, max_length, stride):
              self.input_ids = []
              self.target_ids = []

              token_ids = tokenizer.encode(txt)
              for i in range(0, len(token_ids) - max_length, stride):
                     input_chunk = token_ids[i: i + max_length]
                     target_chunk = token_ids[i+1: i + max_length + 1]
                     # Creating a sliding window of Tensors (vector embeddings)
                     # Where each window is the input and the window + 1 is the target
                     # all index aligned so input aligns with target
                     self.input_ids.append(torch.tensor(input_chunk))
                     self.target_ids.append(torch.tensor(target_chunk))

       def __len__(self):
              return len(self.input_ids)

       def __getitem__(self, idx):
              return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size = 4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

       tokenizer = tiktoken.get_encoding("gpt2")
       dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
       data_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                num_workers=num_workers)

       return data_loader

def test_dataloader_v1(txt, batch_size, max_length, stride, shuffle):

       dataloader = create_dataloader_v1(txt, batch_size, max_length, stride, shuffle)
       data_iter = iter(dataloader)

       first_batch = next(data_iter)
       print(first_batch)

       second_batch = next(data_iter)
       print(second_batch)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
       txt = f.read()


input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 50257
dim_size = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, dim_size)


max_length = 4
data_loader = create_dataloader_v1(txt, batch_size=8,
                                   max_length=max_length,
                                   stride=max_length,
                                   shuffle=False)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print(f"Token IDs: {inputs}")
print(f"Inputs shape: {inputs.shape}")

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, dim_size)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)d