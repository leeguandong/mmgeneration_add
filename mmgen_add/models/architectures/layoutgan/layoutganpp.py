'''
@Time    : 2022/7/12 16:44
@Author  : leeguandon@gmail.com
'''
import torch
import torch.nn as nn
from .layoutgan import TransformerWithToken
from mmgen.models.builder import MODULES
from mmgen.models.architectures.common import get_module_device


@MODULES.register_module()
class LayoutNetppGenerator(nn.Module):
    def __init__(self,
                 dim_latent,
                 num_label,
                 d_model,
                 nhead,
                 num_layers):
        super(LayoutNetppGenerator, self).__init__()
        self.dim_latent = dim_latent

        self.fc_z = nn.Linear(dim_latent, d_model // 2)
        self.emb_label = nn.Embedding(num_label, d_model // 2)
        self.fc_in = nn.Linear(d_model, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, 4)

    def forward(self, noise, label, padding_mask, return_noise=False,**kwargs):
        # noise_batch:4,15,4,在生成bbox_fake
        if noise is None:
            noise_batch = torch.randn(label.size(0), label.size(1), self.dim_latent)

        noise_batch = noise_batch.to(get_module_device(self))

        z = self.fc_z(noise_batch)
        l = self.emb_label(label)
        x = torch.cat([z, l], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        x = torch.sigmoid(x)

        if return_noise:
            return dict(bbox_fake=x, noise_batch=noise_batch)

        return x


@MODULES.register_module()
class LayoutNetppDiscriminator(nn.Module):
    def __init__(self,
                 num_label,
                 d_model,
                 nhead,
                 num_layers,
                 max_bbox):
        """

        Args:
            num_label:
            d_model:
            nhead:
            num_layers:
            max_bbox:
        """
        super(LayoutNetppDiscriminator, self).__init__()

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(d_model=d_model,
                                                    dim_feedforward=d_model // 2,
                                                    nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te,
                                                     num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(self, bbox, label, padding_mask, reconst=False):
        B, N, _ = bbox.size()
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)

        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        x = x[0]

        # logit_disc: [B,]
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc

        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))

            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            x = x.permute(1, 0, 2)[~padding_mask]

            # logit_cls: [M, L]    bbox_pred: [M, 4]
            logit_cls = self.fc_out_cls(x)
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

            return logit_disc, logit_cls, bbox_pred
