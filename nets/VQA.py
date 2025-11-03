import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
from .VIT import Block, VisionTransformer
from facenet_pytorch import InceptionResnetV1
from collections import OrderedDict
from .SwinFace_arch import *


class VITVQA(VisionTransformer):
    def __init__(self, num_score, **kwargs):
        super(VITVQA, self).__init__(**kwargs)
        self.num_score = num_score
        self.head = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.embed_dim, self.embed_dim // 2)),
                ("fc2", nn.Linear(self.embed_dim // 2, self.num_score))
            ]))

    def forward(self, input):
        x = input[0]
        x = self.forward_features(x)
        x = self.head(x)
        return x

class SwinVQA(nn.Module):
    def __init__(self, num_score, embed_dim=512, concat_dim=2112, feature_dim=512, **kwargs):
        super(SwinVQA, self).__init__(**kwargs)
        self.backbone = SwinTransformer()
        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, embed_dim // 2)),
                ("fc2", nn.Linear(embed_dim // 2, num_score))
            ]))
        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        self.head_.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    

    def forward(self, input):
        x = input[0]
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x = self.fam(x)
        x = self.head_(x)
        return x

class VITVQA_mlp(VisionTransformer):
    def __init__(self, num_score, **kwargs):
        super(VITVQA_mlp, self).__init__(**kwargs)
        self.num_score = num_score
        self.head = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.embed_dim, self.embed_dim // 2)),
                ("fc2", nn.Linear(self.embed_dim // 2, self.num_score))
            ]))
        self.mlp = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.num_score, self.num_score * 8)),
                ("fc2", nn.Linear(self.num_score * 8, 1))
            ]))
    def forward(self, input):
        x = input[0]
        x = self.forward_features(x)
        x = self.head(x)
        x_out = self.mlp(x)
        x = torch.cat([x, x_out], dim=1)
        return x
    
class EFFVQA(nn.Module):
    """Efficient Net"""

    def __init__(self, embed_dim, mlp_dim, out_dim):
        super(EFFVQA, self).__init__()
        self.features_extractor = nn.Sequential(*list(models.efficientnet_v2_s(pretrained=True).children())[0][:-1])

        self.pooling = nn.AdaptiveAvgPool1d(1)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)

        self.head1 = nn.Linear(embed_dim, mlp_dim)
        self.head2 = nn.Linear(mlp_dim, out_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = input[0]
        for ii, model in enumerate(self.features_extractor):
            x = model(x)
        x = x.flatten(2) # torch.Size([1, 256, 64])
        x = self.pooling(x) # torch.Size([1, 256, 1])
        x = x.squeeze(2) # torch.Size([1, 256])

        x = self.norm(x)
        x = self.dropout(x)
        # mlp head
        x = self.head1(x)
        # mlp head
        x = self.head2(x)

        return x
    
class GFVQA(nn.Module):
    """Modified ResNet50 for different level feature extraction"""

    def __init__(self, embed_dim, mlp_dim, out_dim):
        super(GFVQA, self).__init__()
        # backbone1 = EfficientNetV2_s
        self.features_extractor = nn.Sequential(*list(models.efficientnet_v2_s(pretrained=True).children())[0][:-1])
        # backbone2 = InceptionResnetV1
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.load_state_dict(torch.load('/mnt/nfs1/hongjiu/pretrain/20180402-114759-vggface2.pt'), strict=False)
        self.pre_feature.eval()    

        # vit fusion
        dpr_1 = [x.item() for x in torch.linspace(0, 0.0, 4)]  # stochastic depth decay rule
        self.confuse_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                  drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=dpr_1[i])
            for i in range(4)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 64 + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.head1 = nn.Linear(embed_dim, mlp_dim)
        self.head2 = nn.Linear(mlp_dim, out_dim)

    def forward(self, input):
        x, y = input
        g = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        for ii, model in enumerate(self.features_extractor):
            x = model(x) # torch.Size([32, 256, 8, 8])

        x = x.flatten(2).transpose(1, 2)  # torch.Size([32, 64, 256])
        g = self.face_down(g) # torch.Size([32, 1, 256])

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # torch.Size([32, 1, 256])
        x = torch.cat((cls_token, x, g), dim=1)  # torch.Size([32, 66, 256])
        x = x + self.pos_embed
        x = self.confuse_blocks(x)
        x = self.norm(x)
        # extract class token
        x = x[:, 0]

        x = self.dropout(x)
        x = self.head1(x)
        x = self.head2(x)

        return x
    
class VITGFVQA(nn.Module):
    def __init__(self, embed_dim, mlp_dim, out_dim, **kwargs):
        super(VITGFVQA, self).__init__(**kwargs)
        # backbone1 = vit_base
        self.features_extractor = VisionTransformer()
        self.features_extractor.load_state_dict(torch.load('/mnt/nfs1/hongjiu/pretrain/jx_vit_base_patch16_224_in21k-e5005f0a.pth'), strict=False)
        # backbone2 = InceptionResnetV1
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.load_state_dict(torch.load('/mnt/nfs1/hongjiu/pretrain/20180402-114759-vggface2.pt'), strict=False)
        self.pre_feature.eval()

        # vit fusion
        dpr_1 = [x.item() for x in torch.linspace(0, 0.0, 4)]  # stochastic depth decay rule
        self.confuse_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                  drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=dpr_1[i])
            for i in range(4)
        ])  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 196 + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.head1 = nn.Linear(embed_dim, mlp_dim)
        self.head2 = nn.Linear(mlp_dim, out_dim)
        
    def forward(self, input):
        x, y = input
        g = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        g = self.face_down(g) # torch.Size([32, 1, 768])
        x = self.features_extractor.forward_patch_features(x) # torch.Size([32, 196, 768])

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # torch.Size([32, 1, 768])
        x = torch.cat((cls_token, x, g), dim=1)  # torch.Size([32, 198, 768])
        x = x + self.pos_embed
        x = self.confuse_blocks(x)
        x = self.norm(x)
        # extract class token
        x = x[:, 0]  # torch.Size([32, 768])

        x = self.dropout(x)
        x = self.head1(x)
        x = self.head2(x)

        return x

class SwinGFVQA(nn.Module):
    def __init__(self, num_score, embed_dim, mlp_dim, concat_dim=2112, feature_dim=512, **kwargs):
        super(SwinGFVQA, self).__init__(**kwargs)
        # backbone1 = Swin_T
        self.backbone = SwinTransformer() # img_size=224, patch_size=2
        # backbone2 = InceptionResnetV1
        self.pre_feature = InceptionResnetV1(pretrained=None)
        self.pre_feature.load_state_dict(torch.load('/mnt/nfs1/hongjiu/pretrain/20180402-114759-vggface2.pt'), strict=False)
        self.pre_feature.eval()

        self.fam = FeatureAttentionNet(in_chans=concat_dim,feature_dim=feature_dim, conv_shared=False,kernel_size=3,
                                                conv_mode="split", channel_attention='CBAM', spatial_attention=None,
                                                pooling="avg")
        self.fam.apply(self._init_weights)
        

        # vit fusion
        dpr_1 = [x.item() for x in torch.linspace(0, 0.0, 2)]  # stochastic depth decay rule
        self.confuse_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                  drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=dpr_1[i])
            for i in range(2)
        ])  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 49 + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.face_down = nn.Linear(512, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(0.2)

        self.head_ = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(embed_dim, mlp_dim)),
                ("fc2", nn.Linear(mlp_dim, num_score))
            ]))
        self.head_.apply(self._init_weights) 

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
        g = self.pre_feature(y).unsqueeze(1) # torch.Size([32, 1, 512])
        g = self.face_down(g) # torch.Size([32, 1, 512])
        local_features, global_features, x = self.backbone(x)
        x = torch.cat([local_features, global_features], dim=1)
        x, _ = self.fam(x) # torch.size([32, 512, 7, 7])
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # torch.Size([32, 1, 512])
        #x = torch.cat((x, g), dim=1)  # torch.Size([32, 50, 512])
        x = torch.cat((cls_token, x, g), dim=1)  # torch.Size([32, 51, 512])
        x = x + self.pos_embed
        x = self.confuse_blocks(x)
        x = self.norm(x)
        # extract class token
        #x = x[:, 0]  # torch.Size([32, 512])
        x = x[:, 1:-1, :]
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.head_(x)

        return x
   
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

