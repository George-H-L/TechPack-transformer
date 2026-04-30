# encoder - 4 layers
# decoder - 4 layers
# 8 heada attention
# 20 mil params

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    # adding positional embeddings using sin cos fucntion
    
    def __init__(self, d_model, max_len=5000, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # making matrix of shape max len and model dimesnions
        pe = torch.zeros(max_len, d_model)
        
        #postion index, returns tensor of dimesnion size one
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #division term and encoding based on odd even 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # adding batch dimension to pe 1, max len, d model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    # adding positional encoding to input embeddings
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    # split inputs into multiple heads and perfom attention in parallel
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # split into multiple heads
        batch_size, seq_length, d_model = x.size()
        #reshape 
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        #transpose to get shape batch size, num heads, seq length, d_k
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        # combine heads back to original shape
        batch_size, num_heads, seq_length, d_k = x.size()
        #transpose back to batch size, seq length, num heads, d_k
        x = x.transpose(1, 2).contiguous()
        #reshape to batch size, seq length, d_model
        return x.view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_length, d_model)
        K = self.W_k(key)
        V = self.W_v(value)  

        # Split into heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)  
        V = self.split_heads(V)  

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_length, seq_length)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Combine heads
        output = self.combine_heads(attended_values)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    # single encoder layer

    # multi head attention, add & norm, feed forward, add & norm
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))  # Add & Norm

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm

        return x

class DecoderLayer(nn.Module):
    # single decoder layer

    # masked multi head attention, add & norm, multi head attention with encoder output, add & norm, feed forward, add & norm
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-attention
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))  # Add & Norm

        # Encoder-Decoder attention
        enc_dec_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attention_output))  # Add & Norm

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # Add & Norm

        return x

class Encoder(nn.Module):
    # stack of encoder layers, needs input embedding positional encoding.

    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_length,dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model

        # input embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        #stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        #embed tokens
        x = self.embedding(src) * math.sqrt(self.d_model)

        #add positional encoding
        x = self.positional_encoding(x)

        #pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x

class Decoder(nn.Module):
    # stack of decoder layers, needs target embedding positional encoding

    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model

        # target embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        #stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        #embed tokens
        x = self.embedding(tgt) * math.sqrt(self.d_model)

        #add positional encoding
        x = self.positional_encoding(x)

        #pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x

class Transformer(nn.Module):
    # complete transformer model with encoder and decoder

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_encoder_layers=4, num_decoder_layers=4, num_heads=8,d_ff=1024, max_seq_length=128, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, num_encoder_layers, d_model, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, num_decoder_layers, d_model, num_heads, d_ff, max_seq_length, dropout)

        #final linear layer to project to vocab
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        # initialising wiht xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # create mask for source sequences (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_length)
        return src_mask

    def make_tgt_mask(self, tgt):
        # create mask for target sequences (padding + subsequent mask)
        tgt_seq_length = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_length)

        # subsequent mask to prevent attending to future tokens
        subsequent_mask = torch.tril(torch.ones((tgt_seq_length, tgt_seq_length), device=tgt.device)).bool()
        tgt_mask = tgt_mask & subsequent_mask.unsqueeze(0)  # (batch_size, 1, tgt_seq_length, tgt_seq_length)

        return tgt_mask

    def forward(self, src, tgt):
        # forward pass, takes source sequence and target seqyence and return output logits

        #create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        #encode source
        encoder_output = self.encoder(src, src_mask)

        #decode target using encoder output
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        #project to vocab
        output_logits = self.output_projection(decoder_output)

        return output_logits

    def generate(self, src, tokenizer, max_length=128, device=None, return_probs=False):
        # greedy decoding - optionally returns per-step softmax probs alongside token ids
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        with torch.no_grad():
            src = src.to(device)
            if src.dim() == 1:
                src = src.unsqueeze(0)

            src_mask = self.make_src_mask(src)
            encoder_output = self.encoder(src, src_mask)

            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), tokenizer.token2id['<SOS>'], dtype=torch.long, device=device)

            token_probs = []

            for _ in range(max_length - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

                logits = self.output_projection(decoder_output[:, -1, :])
                probs = torch.softmax(logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)

                token_probs.append(probs[0, next_token[0, 0]].item())
                tgt = torch.cat([tgt, next_token], dim=1)

                if (next_token == tokenizer.token2id['<EOS>']).all():
                    break

            if return_probs:
                return tgt, token_probs
            return tgt
               
if __name__ == "__main__": 
    # Model parameters
    src_vocab_size = 3547
    tgt_vocab_size = 3547
    d_model = 256
    num_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8
    d_ff = 1024
    max_seq_length = 128
    dropout = 0.1
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    
    #test forward pass with dummy data
    batch_size = 4
    src_len = 10
    tgt_len = 15
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    output_logits = model(src, tgt)
    print("Output logits shape:", output_logits.shape)  # should be (batch_size, tgt_len, tgt_vocab_size)
    print(f"\n🔍 Parameter Breakdown:")
    
    