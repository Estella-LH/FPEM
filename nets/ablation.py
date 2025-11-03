import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
from .VIT import Block, VisionTransformer
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
from .SwinFace_arch import *



class SwinGFVQA_ca(nn.Module):
    def __init__(self, num_score, embed_dim, mlp_dim, concat_dim=2112, feature_dim=512, feat=False, **kwargs):
        super(SwinGFVQA_ca, self).__init__(**kwargs)
        self.backbone = SwinTransformer() 
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.eval()

        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        
        # cross attention block fusion
        self.ca = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)
        #self.ca2 = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)

        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, mlp_dim)),
                ("fc2", nn.Linear(mlp_dim, num_score))
            ]))
        self.head_.apply(self._init_weights)
        self.feat = feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   
        
    def forward(self, input):
        # x, y = input
        x, y = input
        g_ = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        g = self.face_down(g_) # torch.Size([32, 1, 512])
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x, _ = self.fam(x) # torch.size([32, 512, 7, 7])
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        x = self.ca(x, g)
        x = self.norm(x)
        x_mm = x.mean(dim=1)

        x = self.dropout(x_mm)
        x = self.head_(x_mm)

        if self.feat:
            return x, x_mm, g_
        return x


class SwinGFVQA_ca_wo_face(nn.Module):
    def __init__(self, num_score, embed_dim, mlp_dim, concat_dim=2112, feature_dim=512, feat=False, **kwargs):
        super(SwinGFVQA_ca_wo_face, self).__init__(**kwargs)
        self.backbone = SwinTransformer() 
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.eval()

        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        # self.dropout = nn.Dropout(0.2)
        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, mlp_dim)),
                ("fc2", nn.Linear(mlp_dim, num_score))
            ]))
        self.head_.apply(self._init_weights)
        self.feat = feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   
        
    def forward(self, input):
        x, y = input
        # x = input
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x, _ = self.fam(x) # torch.size([32, 512, 7, 7])
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        x = self.norm(x)
        x_mm = x.mean(dim=1)
        x = self.head_(x_mm)

        return x


class SwinGFVQA_add(nn.Module):
    def __init__(self, num_score, embed_dim, mlp_dim, concat_dim=2112, feature_dim=512, feat=False, **kwargs):
        super(SwinGFVQA_add, self).__init__(**kwargs)
        self.backbone = SwinTransformer() 
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.eval()

        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        
        # cross attention block fusion
        # self.ca = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)
        #self.ca2 = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)

        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, mlp_dim)),
                ("fc2", nn.Linear(mlp_dim, num_score))
            ]))
        self.head_.apply(self._init_weights)
        self.feat = feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   
        
    def forward(self, input):
        # x, y = input
        x, y = input
        g_ = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        g = self.face_down(g_) # torch.Size([32, 1, 512])
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x, _ = self.fam(x) # torch.size([32, 512, 7, 7])
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # x = self.ca(x, g)
        x = x + g
        x = self.norm(x)
        x_mm = x.mean(dim=1)

        x = self.dropout(x_mm)
        x = self.head_(x_mm)

        if self.feat:
            return x, x_mm, g_
        return x

class SwinGFVQA_concat(nn.Module):
    def __init__(self, num_score, embed_dim, mlp_dim, concat_dim=2112, feature_dim=512, feat=False, **kwargs):
        super(SwinGFVQA_concat, self).__init__(**kwargs)
        self.backbone = SwinTransformer() 
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.eval()

        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        
        # cross attention block fusion
        # self.ca = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)
        #self.ca2 = CrossAttentionBlock(dim=512, num_heads=4, mlp_dim=2048)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)

        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, mlp_dim)),
                ("fc2", nn.Linear(mlp_dim, num_score))
            ]))
        self.head_.apply(self._init_weights)
        self.feat = feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   
        
    def forward(self, input):
        # x, y = input
        x, y = input
        g_ = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        g = self.face_down(g_) # torch.Size([32, 1, 512])
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x, _ = self.fam(x) # torch.size([32, 512, 7, 7])
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # x = self.ca(x, g)
        x = torch.cat([x, g], dim=1)
        x = self.norm(x)
        x_mm = x.mean(dim=1)

        x = self.dropout(x_mm)
        x = self.head_(x_mm)

        if self.feat:
            return x, x_mm, g_
        return x



if __name__ == '__main__':
    # x = torch.randn(1, 3, 256, 256)
    # y = torch.randn(1, 3, 160, 160)
    # model = GFVQA()
    # z = model(x, y)
    # print(z.shape)

    x = torch.randn(1, 3, 256, 256)
    model = EFFVQA(embed_dim = 256, mlp_dim = 64, out_dim = 1)
    z = model(x)
    print(z.shape)

