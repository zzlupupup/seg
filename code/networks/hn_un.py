import torch
from torch import nn
from networks.vnet import VNet
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import PatchEmbeddingBlock

class CrossAtten(nn.Module):
    def __init__(self, patch_size=[4, 4, 4], fea_dim=128, num_heads=8, dropout=0.1):
        super().__init__()

        self.Q_norm = nn.LayerNorm(fea_dim)
        self.K_norm = nn.LayerNorm(fea_dim)
        self.ff_norm = nn.LayerNorm(fea_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fea_dim // 4, fea_dim),
            nn.Dropout(dropout)
        )

        self.MHA = nn.MultiheadAttention(
            embed_dim=fea_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=patch_size[0], mode='trilinear'),
            nn.Conv3d(fea_dim, fea_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(fea_dim // 2, fea_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(fea_dim // 4),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, Q_emded, context, k_uncertainty_mask):

        Q_norm = self.Q_norm(Q_emded)
        K_norm = self.K_norm(context)
        V = context

        MHA, _ = self.MHA(Q_norm, K_norm, V, need_weights=False, key_padding_mask=k_uncertainty_mask)
        MHA_res = MHA + Q_emded
        tokens = MHA_res + self.ffn(self.ff_norm(MHA_res))
        
        B, N, C = tokens.shape
        patch_dim = round(N ** (1/3)) 
        tokens = tokens.permute(0, 2, 1).contiguous().view(B, C, patch_dim, patch_dim, patch_dim)
        de = self.decoder(tokens)

        return de


class Fusion(nn.Module):
    
    def __init__(self, img_size=[64, 64, 64], patch_size=[4, 4, 4], fea_dim=128, num_heads=8, chan=3):
        super().__init__()

        self.patch_embed = PatchEmbeddingBlock(
            in_channels=chan,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=fea_dim,
            num_heads=num_heads
        )

        self.uncertain_embed = nn.Sequential(
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=1,
                num_heads=1,
                pos_embed_type="none"
            ),
            nn.Sigmoid()
        )

        self.cross_atten = CrossAtten(
            patch_size=patch_size,
            fea_dim=fea_dim,
            num_heads=num_heads,
            dropout=0.1
        )

        out_dim = (fea_dim // 2 )
        self.out = nn.Sequential(
            nn.Conv3d(out_dim, out_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_dim // 2, out_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_dim // 4, chan, kernel_size=1)
        )

    def forward(self, pred_l, pred_r, threshold=None):

        embed_l = self.patch_embed(pred_l)
        embed_r = self.patch_embed(pred_r)
        
        uncertain_l_mask, uncertain_r_mask = None, None
        if threshold:
            pred_l_soft = torch.softmax(pred_l, dim=1)
            pred_r_soft = torch.softmax(pred_r, dim=1)
            
            uncertain_l_map = -torch.sum(pred_l_soft * torch.log(pred_l_soft + 1e-8), dim=1, keepdim=True)
            uncertain_r_map = -torch.sum(pred_r_soft * torch.log(pred_r_soft + 1e-8), dim=1, keepdim=True)

            uncertain_l_seq = self.uncertain_embed(uncertain_l_map).squeeze(-1)
            uncertain_r_seq = self.uncertain_embed(uncertain_r_map).squeeze(-1)

            uncertain_l_mask = (uncertain_l_seq > threshold)
            uncertain_r_mask = (uncertain_r_seq > threshold)


        cross_atten_l = self.cross_atten(embed_l, embed_r, uncertain_r_mask)
        cross_atten_r = self.cross_atten(embed_r, embed_l, uncertain_l_mask)

        cat = torch.cat([cross_atten_l, cross_atten_r], dim=1)
        pred_fusion = self.out(cat)

        return pred_fusion   


class HN(nn.Module):

    def __init__(self, img_size=[64, 64, 64], in_chan=1, out_chan=3, fusion_patch_size=[4, 4, 4], fusion_dim=128, fusion_heads=8):
        super().__init__()
        
        self.model_l = VNet(
            n_channels=in_chan, 
            n_classes=out_chan,
            normalization='batchnorm', 
            has_dropout=True
        )

        self.model_r = SwinUNETR(
            in_channels=in_chan, 
            out_channels=out_chan
        )

        self.fusion = Fusion(
            img_size=img_size,
            patch_size=fusion_patch_size,
            fea_dim=fusion_dim,
            num_heads=fusion_heads,
            chan=out_chan
        )

    def forward(self,x, threshold=None):
        
        pred_l = self.model_l(x)
        pred_r = self.model_r(x)
        pred_fusion = self.fusion(pred_l, pred_r, threshold)

        return pred_fusion, pred_l, pred_r
    

