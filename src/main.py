import torch
import tiktoken
from dataset import GPTDatasetV1
from dataloader import create_dataloader_v1
from model import GPTModel
from text_generator import generate_text_simple
from token_id import text_to_token_ids, token_ids_to_text
from importlib.metadata import version
from download_gpt import download_and_load_gpt2

# Check the versions of the required packages
pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
       ]
print("Package versions:")
for p in pkgs:
    print(f"{p} version: {version(p)}")

# Print the versions of TensorFlow and tqdm
print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))
print("\n")

# Download the model weights for the 124 million parameter model
settings, params = download_and_load_gpt2("124M", "models")
print("Settings:", settings)
print("Params:", params)
print("Parameter dictionary keys:", params.keys())
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# Define the GPT-2 model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# Load data
with open("data/wiz-of-oz.txt", "r", encoding="utf-8") as file:
    txt = file.read()

print(txt[:99])
# txt = input("Enter some text: ")
start_context = input("Enter a starting context: ")

tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(txt)
total_tokens = len(tokenizer.encode(txt))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

dataset = GPTDatasetV1(txt, tokenizer, max_length=4, stride=4)
dataloader = create_dataloader_v1(txt, batch_size=2, max_length=4, stride=4, shuffle=False)
# Future: Figure out how to have a dynamic batch size
# dataloader = create_dataloader_v1(txt, batch_size=8, max_length=256, stride=128, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

batch.append(torch.tensor(tokenizer.encode(start_context)))
batch = torch.stack(batch, dim=0)
print(batch)
torch.manual_seed(123)
gpt = GPTModel(NEW_CONFIG)
gpt.eval();

out = gpt(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))