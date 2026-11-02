"""Training-only dropout with an unchanged inference state-dict inventory."""
import math
from torch.nn import functional as F


def install_dropout(model, probability):
    if not math.isfinite(probability) or not 0<=probability<1:raise ValueError('Invalid dropout')
    if getattr(model,'_training_dropout_installed',False):raise ValueError('Dropout already installed')
    def drop(module, arguments, output):
        return F.dropout(output,p=probability,training=module.training)
    handles=[]
    for block in model.blocks:
        for module in (block.self_attention,block.cross_attention,block.ffn):
            handles.append(module.register_forward_hook(drop))
    model._training_dropout_installed=True
    return handles
