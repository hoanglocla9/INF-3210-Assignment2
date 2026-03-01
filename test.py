from torchao.sparsity.sparse_api import sparsify_, SemiSparseWeightConfig
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

SparseSemiStructuredTensor._FORCE_CUTLASS = True
from torchao.dtypes import SemiSparseLayout
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# Load model and move to GPU
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b")
model = model.half()

sparsify_(model, SemiSparseWeightConfig())

model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
model = model.cuda()
# Prepare input and move to GPU
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Run on GPU
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
next_token_id = outputs.logits[0, -1, :].argmax().item()
next_token = tokenizer.decode([next_token_id])


