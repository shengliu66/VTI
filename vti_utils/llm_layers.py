import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np


class VTILayer(nn.Module):

    def __init__(self, vti_direction, lam):
        super(VTILayer, self).__init__()
        self.vti_direction = vti_direction
        self.lam = lam

    def forward(self, x):
        if self.vti_direction is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            y = 0
            for i in range(len(self.vti_direction)):
                if x.size(1) < 2:
                    lambda_sim = 1.0 #+ torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), -self.vti_direction[i][None,None,:], dim=-1)).unsqueeze(-1)
                    y += self.lam[i] * lambda_sim * F.normalize(self.vti_direction[i], dim=-1).repeat(1,x.shape[1],1)
                else:
                    lambda_sim = 1.0
                    y += self.lam[i] * lambda_sim * F.normalize(self.vti_direction[i], dim=-1)
            y = y/len(self.vti_direction)
            x = F.normalize(F.normalize(x.float(),dim=-1) +  0.1 * y, dim=-1) * norm
                
            return x.half()
        else:
            return x


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_vti_layers(model: PreTrainedModel, vti_drections: Tensor, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(vti_drections) == len(layers)
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, VTILayer(vti_drections[i], alpha)) 

def remove_vti_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        vti_mlp = find_module(layer, mlp_keywords)
        layer.mlp = vti_mlp[0]