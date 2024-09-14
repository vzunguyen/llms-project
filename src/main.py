import torch
import tiktoken
from dataset import GPTDatasetV1
from dataloader import create_dataloader_v1
from model import GPTModel
from text_generator import generate_text_simple
from token_id import text_to_token_ids, token_ids_to_text
from importlib.metadata import version
from download_gpt import download_and_load_gpt2
from loss_calculator import plot_losses
from train_model import train_model_simple

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
print("Parameter dictionary keys:", params.keys())
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# Define the GPT-2 model configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
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
with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
    txt = file.read()

# txt = input("Enter some text: ")
# txt = input("Enter a starting context: ")

tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(txt)
total_tokens = len(tokenizer.encode(txt))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

dataset = GPTDatasetV1(txt, tokenizer, max_length=4, stride=4)
dataloader = create_dataloader_v1(txt, batch_size=2, max_length=4, stride=4, shuffle=False)
# Future: Figure out how to have a dynamic batch size
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

batch.append(torch.tensor(tokenizer.encode(txt)))
batch = torch.stack(batch, dim=0)
print(batch)

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(txt))
train_data = txt[:split_idx]
val_data = txt[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Check that the data was loaded correctly (SIZES)
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
torch.save(model.state_dict(), "model.pth")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)