import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b5, EfficientNet_B5_Weights,
)

from swca_helper import (
    MultiHeadCrossWindowAttention, 
    positional_encoding
)

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone_name, freeze=False):
        super(EncoderBlock, self).__init__()
        self.backbone_name = backbone_name
        backbones = {
            'eff_b5' : efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1),
        }
        self.backbone = backbones.get(backbone_name)
        
        if self.backbone == None:
            print('Check your backbone again ^.^')
            return None
            
        if freeze:
            for v in self.backbone.parameters():
                v.requires_grad = False

    def forward(self, x):
        features = [x]
        
        if self.backbone_name[:3] == 'res': 
            encoder = list(self.backbone.children())
            encoder = torch.nn.Sequential(*(list(encoder)[:-2]))
        else:
            encoder = self.backbone.features
        
        for layer in encoder:
            features.append(layer(features[-1]))

        return features

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, x_channels, skip_channels, desired_channels, layer, window_size, num_heads, 
                    qkv_bias, attn_drop_prob, lin_drop_prob, device):
        super(DecoderBLock, self).__init__()
        self.device = device
        self.attentions = nn.ModuleList()
        for _ in range(layer // 2):
            self.attentions.append(nn.ModuleList([
                MultiHeadCrossWindowAttention(
                    skip_channels=desired_channels, cyclic_shift=False, window_size=window_size, num_heads=num_heads, 
                    qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                ),
                MultiHeadCrossWindowAttention(
                    skip_channels=desired_channels, cyclic_shift=True, window_size=window_size, num_heads=num_heads, 
                    qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                ),
            ]))
        
        self.x_feed2msa = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=desired_channels, stride=1, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(desired_channels),
            nn.LeakyReLU(),
        )
        
        self.skip_feed2msa = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=skip_channels, out_channels=desired_channels, stride=1, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(desired_channels),
            nn.LeakyReLU(),
        )
        
        self.post_msa = nn.Sequential(
            nn.Conv2d(in_channels=desired_channels, out_channels=desired_channels, stride=1, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(desired_channels),
            nn.Sigmoid()
        )
        
        self.early_skip =  nn.Conv2d(
            in_channels=skip_channels, 
            out_channels=desired_channels, 
            stride=1, 
            kernel_size=3, 
            padding='same'
        )
        
        self.x_after_upsample =  nn.Conv2d(
            in_channels=x_channels, 
            out_channels=desired_channels, 
            stride=1, 
            kernel_size=3, 
            padding='same'
        )
        
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=desired_channels*2, out_channels=desired_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(desired_channels),
            nn.LeakyReLU(),
            
            
            nn.Conv2d(in_channels=desired_channels, out_channels=desired_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(desired_channels),
            nn.LeakyReLU(),
        )
        

    def forward(self, skip, x):
        """
        Args: 
            skip    : B, C, H, W
            x       : B, 2C, H/2, W/2
        """
        skip = self.add_abs_pe(skip, self.device)    # B, C, H, W
        x = self.add_abs_pe(x, self.device)          # B, 2C, H/2, W/2
        
        skip_msa = self.skip_feed2msa(skip)  # B, C, H/2, W/2
        x_msa = self.x_feed2msa(x)           # B, C, H/2, W/2
        
        # print(f"[Decoder] skip_msa: {skip_msa.shape}, x_msa : {x_msa.shape}")
        
        for regular_window, shifted_window in self.attentions:
            x_msa = regular_window(skip_msa, x_msa) # B, C, H/2, W/2
            x_msa = shifted_window(skip_msa, x_msa) # B, C, H/2, W/2
        
        post_msa = self.post_msa(x_msa) # B, C, H/2, W/2
        post_msa = F.interpolate(
                        post_msa, 
                        size=[skip.shape[2], skip.shape[3]], 
                        mode='bilinear', 
                        align_corners=True
                    ) # B, C, H, W
        
        skip = self.early_skip(skip) * post_msa
        x = F.interpolate(x, size=[skip.shape[2], skip.shape[3]], mode='bilinear', align_corners=True) # B, 2C, H, W
        x = self.x_after_upsample(x) # B, C, H, W
        out = torch.cat([skip, x], dim=1) # B, 2C, H, W
        out = self.post_process(out) # B, C, H, W
        
        return out

    def add_abs_pe(self, x, device): 
        """
        args:
            x : B, C, H/2, W/2
        return: 
            x : B, C, H/2, W/2
        """
        b, c, origin_h, origin_w = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1) # B, HW/2, C
        b, hw, c = x.shape
        x = x + positional_encoding(max_len=hw, embed_dim=c, device=device) # B, HW/2, C
        x = x.reshape(b, origin_h, origin_w, c).permute(0, 3, 1, 2) # B, C, H/2, W/2
        
        return x

class SWCADepthNet(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone_name, window_sizes:int, explicit_hw:tuple, layers:int, 
                    qkv_bias:bool=True, drop_prob:float=0.15):
        super(SWCADepthNet, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.encoder = EncoderBlock(self.backbone_name, freeze=False).to(device)
        self.explicit_hw = explicit_hw
        dec_heads = 4
        # EfficientNet-B1, B3, B5, B6
        # Todo: EfficientNet Attention Head dimension = 8
        if self.backbone_name == 'eff_b1':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [16, 24, 40, 112, 1280]
        elif self.backbone_name == 'eff_b3':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [24, 32, 48, 136, 1536]
        elif self.backbone_name == 'eff_b5':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [24, 40, 64, 176, 2048]
        elif self.backbone_name == 'eff_b6':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [32, 40, 72, 200, 2304]
        
        else:
            print('Check your backbone again ^.^')
            return None
        
        
        self.decoder = nn.ModuleList([
            DecoderBLock(
                x_channels=features[-1], skip_channels=features[-2], desired_channels=features[-1]//4,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-1]//4, skip_channels=features[-3], desired_channels=features[-1]//8,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-1]//8, skip_channels=features[-4], desired_channels=features[-1]//16,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device
            ),
            DecoderBLock(
                x_channels=features[-1]//16, skip_channels=features[-5], desired_channels=features[-1]//32,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device
            ),
        ]).to(device)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=features[-1]//32, out_channels=1, kernel_size=3, stride=1, padding="same"),
        ).to(device)
        
        
    def forward(self, x):
        real_h, real_w = x.shape[2], x.shape[3]
        x = F.interpolate(
                    x, 
                    size=[self.explicit_hw[0], self.explicit_hw[1]], 
                    mode='bilinear', 
                    align_corners=True
                )
        enc = self.encoder(x) 
        
        if self.backbone_name[:3] == 'res':
            block1 = enc[self.block_idx[0]].clone()
            block2 = enc[self.block_idx[1]].clone()
            block3 = enc[self.block_idx[2]].clone()
            block4 = enc[self.block_idx[3]].clone()
            block5 = enc[self.block_idx[4]].clone()
        else:
            block1 = enc[self.block_idx[0]]
            block2 = enc[self.block_idx[1]]
            block3 = enc[self.block_idx[2]]
            block4 = enc[self.block_idx[3]]
            block5 = enc[self.block_idx[4]]
        
        # print(f"Block 1 : {block1.shape}")
        # print(f"Block 2 : {block2.shape}")
        # print(f"Block 3 : {block3.shape}")
        # print(f"Block 4 : {block4.shape}")
        # print(f"Block 5 : {block5.shape}")
        
        # msa = self.msa(block5)
        # print(f"MSA : {msa.shape}")
        
        u1 = self.decoder[0](block4, block5)
        # print(f"u1 : {u1.shape}")
        u2 = self.decoder[1](block3, u1)
        # print(f"u2 : {u2.shape}")
        u3 = self.decoder[2](block2, u2)
        # print(f"u3 : {u3.shape}")
        u4 = self.decoder[3](block1, u3)
        # print(f"u4 : {u4.shape}")
        
        head = self.head(u4)
        head = F.interpolate(
                    head, 
                    size=[real_h, real_w], 
                    mode='bilinear', 
                    align_corners=True
                )
        
        # print(f"head : {head.shape}")
        
        return head

    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.head]
        for m in modules:
            yield from m.parameters()

if __name__ == '__main__': 
    from torchsummary import summary
    # print(resnet34())
    img = torch.randn((2, 3, 416, 544)).to('cuda')
    model = SWCADepthNet(
        device='cuda', 
        backbone_name='eff_b5', 
        window_sizes=5, 
        explicit_hw=(480, 640),
        layers=2,
    ).to('cuda')
    
                    
    # print(model(img).shape)
    
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, img)
    print('FLOPS TOTAL : ')
    print(flops.total())
    print("===="*5)

    print('FLOPS By Operator : ')
    print(flops.by_operator())
    print("===="*5)
    
    
    # from prettytable import PrettyTable
    # def count_parameters(model):
        
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
        
    #     decoder = 0 
    #     decoder_0 = 0
    #     decoder_1 = 0
    #     decoder_2 = 0
    #     decoder_3 = 0
    #     encoder = 0
    #     dec_attn = 0
        
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad:
    #             continue
    #         params = parameter.numel()
    #         table.add_row([name, f"{params:,}"])
    #         total_params += params
            
    #         splitted_names = name.split('.')
    #         if splitted_names[0] == 'decoder': 
    #             decoder += params
                
    #             if splitted_names[1] == '0': 
    #                 decoder_0 += params
    #             elif splitted_names[1] == '1': 
    #                 decoder_1 += params
    #             if splitted_names[1] == '2': 
    #                 decoder_2 += params
    #             elif splitted_names[1] == '3': 
    #                 decoder_3 += params
                
    #             if splitted_names[2] == 'attentions': 
    #                 dec_attn += params
    #         else:
    #             encoder += params
                
    #     # print(table)
    #     print(f"Total Trainable Params: {total_params:,}")
    #     print(f"Total Encoder Params: {encoder:,}")
    #     print(f"Total Decoder Params: {decoder:,}")
    #     print(f"Total Decoder 0 : {decoder_0:,} | 1 : {decoder_1:,} | 2 : {decoder_2:,} | 3 : {decoder_3:,}")
    #     print(f"Total Attention : {dec_attn:,}")
        
    #     return total_params
    
    # count_parameters(model)