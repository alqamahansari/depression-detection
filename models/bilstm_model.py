import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, dropout=0.5):
        super(BiLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_cat = self.dropout(hidden_cat)

        out = self.fc(hidden_cat)
        return self.sigmoid(out)