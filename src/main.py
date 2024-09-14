import torch
import tiktoken
from dataset import GPTDatasetV1
from dataloader import create_dataloader_v1
from model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

with open("data/the-verdict.txt", "r") as f:
    txt = f.read()

# txt = input("Enter some text: ")

tokenizer = tiktoken.get_encoding("gpt2")

dataset = GPTDatasetV1(txt, tokenizer, max_length=4, stride=4)

dataloader = create_dataloader_v1(txt, batch_size=8, max_length=4, stride=4, shuffle=False)
# Future: Figure out how to have a dynamic batch size
# dataloader = create_dataloader_v1(txt, batch_size=8, max_length=256, stride=128, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)