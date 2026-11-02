"""Expand trained geometric scoring into an equivalent unconstrained classifier.

This changes only the output-head parameterization. It does not convert an
autoregressive model to NCM. Floating-point reassociation can cause small errors.
"""
import torch


@torch.no_grad()
def initialize_from_geometry(target,source):
    if target.input_norm!=source.input_norm:raise ValueError('Input normalization differs')
    if len(target.classifiers)!=len(source.projections):raise ValueError('Level inventory differs')
    for i,(classifier,projection,bias) in enumerate(zip(target.classifiers,source.projections,source.biases)):
        book=getattr(source,f'book_{i}').to(classifier.weight)
        projection_weight=projection.weight.to(classifier.weight)
        if classifier.weight.shape!=(book.shape[0],projection_weight.shape[1]):
            raise ValueError('Classifier dimensions differ')
        square=book.square().sum(-1)
        scale=source.log_temperature[i].clamp(-8,6).exp().to(square)/square.mean().clamp_min(1e-6)
        classifier.weight.copy_((2*scale*book)@projection_weight)
        offset=-square
        if projection.bias is not None:offset=offset+2*book@projection.bias.to(book)
        classifier.bias.copy_(scale*offset+bias.to(square))
    return target
