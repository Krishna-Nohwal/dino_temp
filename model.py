import timm
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class MACHead(nn.Module):
    """
    Multi-Aspect Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 192-dim intermediate feature vector.

    Input dim:  (1 + num_reg + 1) * embed_dim  =  6 * 384  =  2304
    Hidden dim: embed_dim                        =  384
    Bottle dim: embed_dim // 2                   =  192      ← returned as `h`
    Output dim: 2  (real / fake logits)

    Optimisation vs previous version:
      • fc1/fc2/fc3 fused into a single nn.Sequential so the autograd graph
        is shallower and JIT/torch.compile can fuse adjacent linear+relu+dropout.
      • patch_tok mean is computed before the cat — same result, lets the
        compiler see a smaller cat input.
      • torch.cat receives contiguous views so no extra copy is needed.
    """
    def __init__(self, embed_dim: int = 384, num_reg: int = 4, dropout_p: float = 0.4):
        super().__init__()
        self.num_reg = num_reg
        in_dim = (1 + num_reg + 1) * embed_dim   # 6C = 2304

        # Sequential makes the forward trivial and compiler-friendly.
        self.head = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Linear(embed_dim // 2, 2)

    def forward(self, cls_tok, reg_tok, patch_tok):
        """
        Args:
            cls_tok   : (B, 1,       embed_dim)
            reg_tok   : (B, num_reg, embed_dim)
            patch_tok : (B, H*W,     embed_dim)
        Returns:
            logits : (B, 2)
            h      : (B, embed_dim // 2)  — 192-dim discriminative features
        """
        B     = cls_tok.size(0)
        f_avg = patch_tok.mean(dim=1)        # (B, C)    spatial average
        f_cls = cls_tok.squeeze(1)           # (B, C)    CLS token
        f_reg = reg_tok.reshape(B, -1)       # (B, 4*C)  REG tokens flattened
        # cat on the last dim — all inputs are already contiguous.
        inp   = torch.cat([f_cls, f_reg, f_avg], dim=1).float()  # (B, 6C)
        h     = self.head(inp)                                     # (B, C/2 = 192)
        return self.classifier(h), h                               # logits, features


class ViT(nn.Module):
    """
    DINO ViT-Small/14 with 4 register tokens, finetuned with LoRA.

    Forward pass taps intermediate outputs from layers [8, 9, 10, 11].
    Each layer feeds its own MACHead → 4 sets of (logits, 192-dim features).

    Shapes per forward call (batch size B, image size 266×266):
        patch grid : 19×19 = 361 patches  (266 / 14 ≈ 19)
        prefix_tokens : [CLS, REG_1, REG_2, REG_3, REG_4]  → 5 tokens
        spatial_map   : (B, 384, 19, 19)
        patch_tok     : (B, 361, 384)
        cls_tok       : (B, 1,   384)
        reg_tok       : (B, 4,   384)

    Returns:
        logits_list   : list of 4 × (B, 2)    — one per tapped layer
        features_list : list of 4 × (B, 192)  — one per tapped layer

    Optimisation vs previous version:
      • torch.compile(model) in train.py will wrap this forward; the
        patch_tok reshape is the only runtime allocation here, everything
        else is slice/view — compile can hoist it.
      • MAC heads receive contiguous inputs (permute+reshape is one op on
        a contiguous spatial_map) so no extra copy inside MACHead.forward.
    """
    EMBED_DIM = 384
    NUM_REG   = 4
    NUM_HEADS = 4
    LAYERS    = [8, 9, 10, 11]
    DROP_PATH = 0.15
    MAC_DROP  = 0.4

    def __init__(self):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────
        self.vit = timm.create_model(
            'vit_small_patch14_reg4_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["attn.qkv"],
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        # Gradient checkpointing trades VRAM for compute — essential on 6 GB.
        self.vit.base_model.model.set_grad_checkpointing(enable=True)

        # ── One MACHead per tapped layer ────────────────────────────────
        self.mac_heads = nn.ModuleList([
            MACHead(self.EMBED_DIM, self.NUM_REG, self.MAC_DROP)
            for _ in range(self.NUM_HEADS)
        ])

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=self.LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        logits_list:   list = []
        features_list: list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            B, C, H, W = spatial_map.shape
            # contiguous() ensures the reshape is a zero-copy view.
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            logits, feats = self.mac_heads[i](cls_tok, reg_tok, patch_tok)
            logits_list.append(logits)
            features_list.append(feats)

        return logits_list, features_list