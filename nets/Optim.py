import torch
import torch.nn as nn
from torch.optim import AdamW

class PreciseAdjustAdamW(AdamW):
    def __init__(self, clip, **kwargs):
        super(PreciseAdjustAdamW, self).__init__(**kwargs)
        self.clip = clip

    def step(self, closure=None):

        self.convert_model_to_fp32(self.clip)
        super().step(closure=closure)
        self.convert_model_to_fp16(self.clip)


    def convert_model_to_fp32(self, model: nn.Module):
        for p in model.parameters():
            p.data = p.data.float()
            if p.grad is not None:
                p.grad.data = p.grad.data.float()

    def convert_model_to_fp16(self, model: nn.Module):
        """Convert applicable model parameters to fp16"""

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()
            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()
            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        model.apply(_convert_weights_to_fp16)
