from modelling.mistral import UnmaskMistralModel
import torch

model = UnmaskMistralModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)