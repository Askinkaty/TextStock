import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


#1. Self-Attention (within the tweets of a day):
# This type of attention allows each tweet embedding to attend to other tweet embeddings within the same day.
# Tweets that are more informative or relevant to the overall sentiment of the day will receive higher attention weights

class HeadlineAggregatorWithSelfAttention(nn.Module):
    def __init__(self, bert_output_dim, lstm_hidden_dim, num_layers=1, bidirectional=False, attention_heads=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=bert_output_dim, hidden_size=lstm_hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.final_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.attention_heads = attention_heads
        self.attention_dim = bert_output_dim  # You can adjust this

        self.query = nn.Linear(bert_output_dim, self.attention_dim)
        self.key = nn.Linear(bert_output_dim, self.attention_dim)
        self.value = nn.Linear(bert_output_dim, self.attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(self.attention_dim, bert_output_dim) # Project back to original dim


def forward(self, headline_embeddings):
     # headline_embeddings: (batch_size, num_headlines, bert_output_dim)
    batch_size, num_headlines, _ = headline_embeddings.size()

     # Attention mechanism
    Q = self.query(headline_embeddings)  # (batch_size, num_headlines, attention_dim)
    K = self.key(headline_embeddings)    # (batch_size, num_headlines, attention_dim)
    V = self.value(headline_embeddings)  # (batch_size, num_headlines, attention_dim)

     # Calculate attention scores
    attention_scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32)) # Scaled dot-product
    attention_weights = self.softmax(attention_scores) # (batch_size, num_headlines, num_headlines)

     # Apply attention weights to values
    weighted_embeddings = torch.matmul(attention_weights, V) # (batch_size, num_headlines, attention_dim)
    weighted_embeddings = self.output_projection(weighted_embeddings) # Project back

     # Aggregate using LSTM on the weighted embeddings
    h0 = torch.zeros(self.lstm.num_layers * (2 if self.bidirectional else 1), batch_size, self.lstm.hidden_size).to(headline_embeddings.device)
    c0 = torch.zeros(self.lstm.num_layers * (2 if self.bidirectional else 1), batch_size, self.lstm.hidden_size).to(headline_embeddings.device)

    output, (h_n, c_n) = self.lstm(weighted_embeddings, (h0, c0))

    if self.bidirectional:
        final_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
    else:
        final_state = h_n[-1]

    return final_state



# 2. Attention over LSTM Hidden States:
# Instead of applying attention directly to the tweet embeddings,
# you could first pass the tweet embeddings through an LSTM and then apply attention to the hidden states of that LSTM.
# This allows the LSTM to capture some sequential information within the tweets before the attention mechanism focuses on the most important hidden states.

class HeadlineAggregatorWithLSTMAttention(nn.Module):
    def __init__(self, bert_output_dim, lstm_hidden_dim, attention_dim):
        super().__init__()
        self.tweet_lstm = nn.LSTM(input_size=bert_output_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.attention_query = nn.Linear(lstm_hidden_dim, attention_dim)
        self.attention_keys = nn.Linear(lstm_hidden_dim, attention_dim)
        self.attention_values = nn.Linear(lstm_hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, headline_embeddings):
        # headline_embeddings: (batch_size, num_headlines, bert_output_dim)
        batch_size, num_headlines, _ = headline_embeddings.size()

         # LSTM to process tweet embeddings
        lstm_out, _ = self.tweet_lstm(headline_embeddings)  # (batch_size, num_headlines, lstm_hidden_dim)

         # Attention mechanism
        queries = self.attention_query(lstm_out)  # (batch_size, num_headlines, attention_dim)
        keys = self.attention_keys(lstm_out)  # (batch_size, num_headlines, attention_dim)
        attention_logits = self.attention_values(torch.tanh(queries + keys)).squeeze(-1)  # (batch_size, num_headlines)
        attention_weights = self.softmax(attention_logits)  # (batch_size, num_headlines)

         # Weighted sum of LSTM outputs
        aggregated_embedding = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, lstm_hidden_dim)

        return aggregated_embedding