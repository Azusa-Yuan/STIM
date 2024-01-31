import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import copy

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class backbone(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        return x


class spatial_extractor(nn.Module):
    def __init__(self, node_feats, embedding_dim=32, if_sample=False,
                 expand=False, dropout=0.15, pooling_size=1, pooling_mode=None, channel=1):
        super().__init__()

        self.channel = channel
        # 只用了第一个feature， train_length, node, feature
        self.node_feats = torch.Tensor(node_feats[..., range(self.channel)])
        train_length, num_nodes, feature = node_feats.shape
        self.pool_kernel_size = pooling_size
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.train_length = train_length

        if pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pool_kernel_size,
                                              stride=self.pool_kernel_size)
        elif pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pool_kernel_size,
                                              stride=self.pool_kernel_size)
        else:
            pooling_size = 1
            self.pooling_layer = None

        self.dim_fc = 608

        if self.pooling_layer is not None:
            self.node_feats = self.pooling_layer(self.node_feats.transpose(0, 2)).transpose(0, 2)

        # network structure
        self.conv1 = torch.nn.Conv1d(1, 32, 12, stride=1, dilation=2)  # 1 hour --> per timestep
        self.bn1 = torch.nn.BatchNorm1d(32)

        self.conv2 = torch.nn.Conv1d(32, 32, 12, stride=12, dilation=2)  # 2 hour --> per hour
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.conv3 = torch.nn.Conv1d(32, 32, 24, stride=24, dilation=2)  # 1 day --> per day
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.expand = expand
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.embedding_dim)

        self.power = nn.Parameter(torch.ones(1))


    def forward(self, batch_size, epoch=None):
        """
            output: N, embedding_dim or B, N, embedding_dim
        """
        # time_series_emb B embed_dim N 1
        # train_length, node→ node, 1, train_length,
        x = self.node_feats.permute(1, 2, 0).to(self.power.device)
        # node, 1, train_length → node, 8, train_length-9 → node, 16, train_length-18
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = self.bn3(F.leaky_relu(self.conv3(x)))


        x = x.view(self.num_nodes, -1)
        x = self.bn(F.relu(self.dropout(self.fc(x))))

        if self.expand:
            # node, embedding_dim → batch, node, embedding_dim
            x = x.unsqueeze(0).expand(batch_size, self.num_nodes, -1)

        return x

def moving_avg(data, kernel_size, stride=1):
    # padding on the both ends of time series
    # data = [B, L, N, C] 长度 节点数 通道数
    x = data[:, :, :, 0]
    # B, L, N
    # 使用首个数据填充前面的padding 使用最后一个数据填充前面的padding
    front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    moving_mean = torch.cat([front, x, end], dim=1)
    moving_mean = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)(moving_mean.permute(0, 2, 1))
    moving_mean = moving_mean.permute(0, 2, 1)
    data[:, :, :, 0] = moving_mean
    res = x - moving_mean
    resData = copy.deepcopy(data)
    resData[:, :, :, 0] = res
    return resData, data


class STIF(nn.Module):
    def __init__(self, input_dim, input_embedding_dim, steps_per_day, tod_embedding_dim,
                 dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim, node_feats,
                 in_steps, num_nodes, decomposition=False):
        super().__init__()
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.in_steps = in_steps
        self.decomposition = decomposition

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if self.decomposition:
            self.eTrend = nn.Linear(1, input_embedding_dim)
            self.ePeriodic = nn.Linear(1, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
            self.longtime_embedding = spatial_extractor(node_feats, embedding_dim=adaptive_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

    def forward(self, x):
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., -2]
        if self.dow_embedding_dim > 0:
            dow = x[..., -1]
        x = x[..., : self.input_dim]

        if self.decomposition:
            trend, res = moving_avg(x, kernel_size=3)
            trend = self.eTrend(trend)
            res = self.ePeriodic(res)
            x = trend + res
        else:
            x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            long_time_embedding = self.longtime_embedding(64)
            spatial_emb = long_time_embedding.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        # (batch_size, in_steps, num_nodes, dim)
        x = torch.cat(features, dim=-1)
        return x


class STIM(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
            node_feats=None
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.init_embedding = STIF(input_dim, input_embedding_dim, steps_per_day, tod_embedding_dim,
                                   dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim, node_feats,
                                   in_steps, num_nodes)

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.backbone = backbone(model_dim=self.model_dim, feed_forward_dim=feed_forward_dim,
                                          num_heads=num_heads, dropout=dropout, num_layers=num_layers)

    def forward(self, history_data, future_data=None, batch_seen=None, epoch=None, train=None):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size = x.shape[0]

        x = self.init_embedding(x)

        x = self.backbone(x)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
