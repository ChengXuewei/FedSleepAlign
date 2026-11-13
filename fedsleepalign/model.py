"""The sleep staging network: an epoch encoder (two residual convolutional blocks with BN, a bidirectional GRU and a
Transformer encoder averaged over time) and a sequence encoder (bidirectional LSTM) over the L epochs of a window,
followed by a two-layer head. All BN layers lie in the residual blocks (model.cnn)."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Convolution (stride 1, length-preserving zero padding), BN, ReLU, convolution of length 3, BN, plus a shortcut
    of a 1x1 convolution and BN, then ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(out + self.downsample(x))


class SleepStageModel(nn.Module):
    """Epoch encoder and head; SeqCtxModel adds the sequence encoder between them."""
    def __init__(self, num_classes=5, dropout_rate=0.4, nodes=128):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualBlock(3, nodes, kernel_size=49), nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(dropout_rate),
            ResidualBlock(nodes, nodes * 2, kernel_size=25), nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout_rate))
        self.gru = nn.GRU(input_size=nodes * 2, hidden_size=nodes * 2, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=nodes * 4, nhead=8, dim_feedforward=nodes * 8, dropout=dropout_rate,
                                       batch_first=True), num_layers=4)
        self.fc = nn.Sequential(nn.Linear(nodes * 4, nodes * 2), nn.ReLU(), nn.Dropout(dropout_rate),
                                nn.Linear(nodes * 2, num_classes))


class SeqCtxModel(nn.Module):
    """Input (B, 3, L, 3000): B windows of L epochs; output stage logits (B, 5, L)."""
    def __init__(self, seqlen, num_classes=5, dropout_rate=0.4, nodes=128):
        super().__init__()
        base = SleepStageModel(num_classes, dropout_rate, nodes)
        self.cnn, self.gru, self.transformer, self.fc = base.cnn, base.gru, base.transformer, base.fc
        self.seqlen = seqlen
        self.seq_lstm = nn.LSTM(input_size=nodes * 4, hidden_size=nodes * 2, num_layers=1, batch_first=True,
                                bidirectional=True)

    def encode_epoch(self, x):                                   # (N, 3, 3000) -> (N, 512)
        x, _ = self.gru(self.cnn(x).permute(0, 2, 1))
        return self.transformer(x).mean(dim=1)

    def forward(self, x):
        B, C, W, L = x.shape
        emb = self.encode_epoch(x.permute(0, 2, 1, 3).reshape(B * W, C, L)).view(B, W, -1)
        ctx, _ = self.seq_lstm(emb)
        return self.fc(ctx.reshape(B * W, -1)).view(B, W, -1).permute(0, 2, 1)


class HybridBN(nn.Module):
    """Hybrid BN (Chen et al., ICML 2025) for (N, C, L) inputs, following the official FedHBN code: training normalizes
    with a * batch + (1 - a) * global moments, a = sigmoid(factor) per channel; while collecting, the mixed moments of
    successive batches are merged by their counts; evaluation uses the global moments."""
    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps, self.collect = eps, False
        self.factor = nn.Parameter(torch.zeros(c))
        self.weight, self.bias = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))
        for name, v in (("global_mean", 0), ("global_var", 1), ("running_mean", 0), ("running_var", 0)):
            self.register_buffer(name, torch.full((c,), float(v)))
        self.register_buffer("running_num", torch.zeros(1))

    def forward(self, x):
        if self.training:
            a = torch.sigmoid(self.factor)
            mean = (1 - a) * self.global_mean + a * x.mean(dim=(0, 2))
            var = (1 - a) * self.global_var + a * x.var(dim=(0, 2), unbiased=False)
            if self.collect:
                n0, n = self.running_num.clone(), x.shape[0]
                m = (n0 * self.running_mean + n * mean) / (n0 + n)
                self.running_var.copy_((n0 * self.running_var + n * var + n * (mean - m) ** 2) / (n0 + n))
                self.running_mean.copy_(m); self.running_num.add_(n)
        else:
            mean, var = self.global_mean, self.global_var
        return (x - mean[:, None]) / (var[:, None] + self.eps).sqrt() * self.weight[:, None] + self.bias[:, None]


def build_model(scheme="fedbn", seqlen=20):
    model = SeqCtxModel(seqlen)
    if scheme == "hbn":                                          # the six BN layers become hybrid BN layers
        for blk in (model.cnn[0], model.cnn[3]):
            blk.bn1, blk.bn2 = HybridBN(blk.bn1.num_features), HybridBN(blk.bn2.num_features)
            blk.downsample[1] = HybridBN(blk.downsample[1].num_features)
    return model
