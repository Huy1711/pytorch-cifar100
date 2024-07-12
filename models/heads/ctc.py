import torch.nn as nn
import torch.nn.functional as F

class CTCHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim=512, 
            num_layers=2
        ):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, None))

        # seq2seq decode cangjie model
        self.rnn = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True
        )

        self.embedding = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out = self.avg(x)
        b, c, h, w = out.size()
        # assert h == 1, "the height of output must be 1"
        out = out.squeeze(2)
        out = out.permute(2, 0, 1)  # [w, b, c]

        out, _ = self.rnn(out)
        out = self.embedding(out)

        out = F.log_softmax(out, dim=2)
        return out

