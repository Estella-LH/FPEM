import torch
import clip
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Normalize
import sys, os
from collections import OrderedDict
from functools import partial
current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir_path)
from nets.VQA import SwinGFVQA_ca
from nets.SwinFace_arch import CrossModalBlock

ImageFile.LOAD_TRUNCATED_IMAGES = True

attractiveness = ['bad', 'poor', 'fair', 'good', 'perfect']

   
class FACLIP(nn.Module):
    """
    Facial Attractiveness CLIP (FACLIP)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.cuda.current_device()
        self.clip, _ = clip.load("ViT-B/16", device=self.device, jit=False)
        for p in self.clip.token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip.transformer.parameters():
            p.requires_grad = False
        self.clip.positional_embedding.requires_grad = False
        self.clip.text_projection.requires_grad = False
        for p in self.clip.ln_final.parameters():
            p.requires_grad = False

        self.texts = torch.cat(
            [clip.tokenize(f"a photo of a person with {a} attractiveness") for a in attractiveness]).to(self.device)
    
    def get_embeddings(self, img, texts):
        x = self.clip.encode_image(img)
        text = self.clip.encode_text(texts) # torch.Size([5, 512])
        return x, text

    def get_score_by_similarity(self, x, text):
        logit_scale = self.clip.logit_scale.exp() # tensor(100.)

        x = x / x.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        if len(text.shape) == 3:
            text = text.permute(0, 2, 1)
            # if self.device!="cpu":
            #     text = text.half()
            #text = text.half()
        else:
            text = text.t()
        logits_per_image = logit_scale*x @ text
        logits_quality = F.softmax(logits_per_image, dim=-1)
        logits_quality = logits_quality.squeeze()
        # logits_quality = logits_quality.unsqueeze(0)

        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        
        return quality, logits_quality

    def forward(self, input):
        x, x1, x2 = input
        x, text = self.get_embeddings(x, self.texts)
        quality, logits_quality = self.get_score_by_similarity(x, text)
        quality = quality.unsqueeze(-1)
        return tuple([quality, logits_quality])


class SKNet(nn.Module):
    def __init__(self, features, M=2, r=16, L=32, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.features = features
        d = max(int(features/r), L)
        self.fc = nn.Sequential(nn.Linear(features, d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, feat1, feat2):
        feats = torch.cat([feat1, feat2], dim=1)
        batch_size = feat1.shape[0]
        feats_S = feat1 + feat2
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V.unsqueeze(1)
        

class FPEM(nn.Module):
    """
    Facial Prior Enhanced Multimodal Model (FPEM)
    """
    def __init__(self,
                 sw_num_score=1, sw_embed_dim=512, sw_mlp_dim=64, sw_feat=True,
                 ca_dim=512, ca_num_heads=4, ca_mlp_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.cuda.current_device()

        #models
        self.FACLIP = FACLIP()
        self.SwinGFVQA_ca = SwinGFVQA_ca(num_score=sw_num_score, embed_dim=sw_embed_dim, mlp_dim=sw_mlp_dim, feat=sw_feat)
        #self.ca = CrossModalBlock(dim=ca_dim, num_heads=ca_num_heads, mlp_dim=ca_mlp_dim)
        self.texts = torch.cat([clip.tokenize(f"a photo of a person with {a} attractiveness") for a in attractiveness]).to(self.device)
        self.late_fusion = nn.Sequential(OrderedDict([
                ("lf1", nn.Linear(3, 3)),
                ("lf2", nn.Linear(3, 1))
            ]))
        self.late_fusion.apply(self._init_weights) 
        # self.linear = nn.Linear(512, 512)
        self.fuse = SKNet(512)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1/3)

    def forward(self, input):
        x, x1, x2 = input
        batch_size = x.size(0)
        score1, logits_swin_ca_, g_ = self.SwinGFVQA_ca([x1, x2])  # torch.Size([32, 512])
        logits_swin_ca = logits_swin_ca_.unsqueeze(1)  # torch.Size([32, 1, 512])
        score1 = score1.squeeze(1)

        x_, text_ = self.FACLIP.get_embeddings(x, self.texts)

        x = x_.unsqueeze(1)  # torch.Size([32, 1, 512])
        text = text_.unsqueeze(0).expand(batch_size, -1, -1)  # torch.Size([32, 5, 512])
        # multi-modal fusion
        # text = self.ca(text.float(), logits_swin_ca)  # torch.Size([32, 5, 512])
        # add on textual embedding
        # text = text.float() + logits_swin_ca
        # text = self.linear(text)
        # add on visual embedding
        # x = torch.concat([x.float(), logits_swin_ca], dim=-1)
        # x = x.float() + logits_swin_ca
        # x = self.linear(x)
        # text = text.float()
        # fuse with sknet with visual embedding
        x = self.fuse(x, logits_swin_ca)
        text = text.float()
        
        score2, _ = self.FACLIP.get_score_by_similarity(x_, text_)

        score3, _ = self.FACLIP.get_score_by_similarity(x, text)

        return tuple([score3.unsqueeze(-1), _])
        #return score3.unsqueeze(-1)


class Clip_visual(nn.Module):
    """
    CLIP visual encoder + linear
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.cuda.current_device()
        self.clip, _ = clip.load("ViT-B/16", device=self.device, jit=False)
        self.clip = self.clip.float()
        for p in self.clip.token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip.transformer.parameters():
            p.requires_grad = False
        self.clip.positional_embedding.requires_grad = False
        self.clip.text_projection.requires_grad = False
        for p in self.clip.ln_final.parameters():
            p.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.texts = torch.cat(
            [clip.tokenize(f"a photo of a person with {a} attractiveness") for a in attractiveness]).to(self.device)

    
    def get_embeddings(self, img, texts):
        x = self.clip.encode_image(img)
        text = self.clip.encode_text(texts) # torch.Size([5, 512])
        return x, text
    
    def forward(self, input):
        x = input[0]
        x, text = self.get_embeddings(x, self.texts)
        x = self.mlp(x)
        
        return x



if __name__ == '__main__':
    device='cpu'
    fpem = FPEM()
    x = torch.randn(32, 3, 224, 224).to(device)
    x1 = torch.randn(32, 3, 112, 112).to(device)
    x2 = torch.randn(32, 3, 160, 160).to(device)
    fpem.forward((x,x1,x2))