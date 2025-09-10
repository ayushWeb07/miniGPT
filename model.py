# import packages
import torch
from torch import nn
import math

# constants
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 128, # Context length
    "emb_dim": 384,         # Embedding dimension
    "n_heads": 6,          # Number of attention heads
    "n_layers": 6,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")



# multi head attention: causal + mha
class MultiHeadAttention(nn.Module):
    
    def __init__(self, config: dict= GPT_CONFIG_124M):
        super().__init__()
        
        self.config= config
        
        self.dropout= nn.Dropout(config["drop_rate"])
        
        # init. the weight matrices
        self.w_q= nn.Linear(config["emb_dim"], config["emb_dim"], bias= config["qkv_bias"])
        self.w_k= nn.Linear(config["emb_dim"], config["emb_dim"], bias= config["qkv_bias"])
        self.w_v= nn.Linear(config["emb_dim"], config["emb_dim"], bias= config["qkv_bias"])
        self.w_o= nn.Linear(config["emb_dim"], config["emb_dim"])
        
        
        self.d_model= config["emb_dim"]
        self.n_heads= config["n_heads"]
        self.d_head= self.d_model // self.n_heads
        
        self.register_buffer("mask", torch.tril(torch.ones((config["context_length"], config["context_length"]))))
    
    def forward(self, x: torch.Tensor):
        # x -> (B, S, d_m)
        b, s, d_m= x.shape
        
        # calc the query, key, value vectors -> (B, S, d_m)
        q= self.w_q(x)
        k= self.w_k(x)
        v= self.w_v(x)
        
        # change shape of query, key, value vectors: (B, S, d_m) -> (B, S, n_h, d_h)
        q= q.view(b, s, self.n_heads, self.d_head)
        k= k.view(b, s, self.n_heads, self.d_head)
        v= v.view(b, s, self.n_heads, self.d_head)
        
        # change shape of query, key, value vectors: (B, S, n_h, d_h) -> (B, n_h, S, d_h)
        q= q.transpose(1, 2)
        k= k.transpose(1, 2)
        v= v.transpose(1, 2)
        
        
        # calc the attention score
        atten_scores= q @ k.transpose(-1, -2) # (B, n_h, S, d_h) @ (B, n_h, d_h, S) -> (B, n_h, S, S)
        atten_scores/= self.d_head ** 0.5 # scale the attention score by root of d_m
        
        # apply the mask
        mask= self.mask[:s, :s]
        atten_scores.masked_fill_(mask == 0, float("-inf"))
        
        # calc the attention weights
        atten_weights= torch.softmax(atten_scores, dim= -1) # (B, n_h, S, S)
        atten_weights= self.dropout(atten_weights)
        
        # calc the output
        out= atten_weights @ v # (B, n_h, S, S) @ (B, n_h, S, d_h) -> (B, n_h, S, d_h)
        
        # change shape of the output: (B, n_h, S, d_h) -> (B, S, n_h, d_h) -> (B, S, d_m)
        out= out.transpose(1, 2)
        out= out.contiguous().view(b, s, self.d_model)
        
        # pass the output through the final output weight matrix
        final_out= self.w_o(out) # (B, S, d_m)
        
        return final_out




# feed forward mlp
class FeedForwardMLP(nn.Module):
    
    def __init__(self, config: dict= GPT_CONFIG_124M):
        
        super().__init__()
        
        
        # mlp sequence
        self.mlp= nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["emb_dim"] * 4, config["emb_dim"])
        )
    
    def forward(self, x: torch.Tensor):
        out= self.mlp(x)
        return out
    



# transformer block
class TransformerBlock(nn.Module):
    
    def __init__(self, config: dict= GPT_CONFIG_124M):
        
        super().__init__()
        
        # mha
        self.norm1= nn.RMSNorm(normalized_shape= config["emb_dim"])
        self.mha= MultiHeadAttention(config)
        
        # mlp
        self.norm2= nn.RMSNorm(normalized_shape= config["emb_dim"])
        self.mlp= FeedForwardMLP(config)
        
        # dropout
        self.dropout= nn.Dropout(config["drop_rate"])
        
    def forward(self, x: torch.Tensor):
        
        # first part of the transformer block
        shortcut= x
        x= self.norm1(x)
        x= self.mha(x)
        x= self.dropout(x)
        x= x + shortcut
        
        
        # second part of the transformer block
        shortcut= x
        x= self.norm2(x)
        x= self.mlp(x)
        x= self.dropout(x)
        x= x + shortcut
        
        
        return x
    
    
    
    
# whole GPT arch
class GPT(nn.Module):
    
    def __init__(self, config: dict= GPT_CONFIG_124M):
        
        super().__init__()
        
        # input layers
        self.token_embed_layer= nn.Embedding(num_embeddings= config["vocab_size"], embedding_dim= config["emb_dim"])
        
        self.pos_embed_layer= nn.Embedding(num_embeddings= config["context_length"], embedding_dim= config["emb_dim"])
        
        self.transformer_blocks= nn.ModuleList([
            TransformerBlock(config) for _ in range(config["n_layers"])
        ])
        
        self.final_norm= nn.RMSNorm(normalized_shape= config["emb_dim"])
        
        self.final_output_projection= nn.Linear(config["emb_dim"], config["vocab_size"], bias= False)
        
        # weight tying
        self.final_output_projection.weight= self.token_embed_layer.weight
        
        # weight initialization
        self.apply(lambda m: self._init_weights(m))
        
        for pn, p in self.named_parameters():
            if pn.endswith('w_o.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layers"]))
        
        
    def forward(self, x: torch.Tensor):
        
        # x -> (B, S)
        b, s= x.shape
        
        # pass through the input layers
        x= self.token_embed_layer(x) # (B, S, d_m)
        pos= self.pos_embed_layer(torch.arange(s, device= DEVICE)) # (S, d_m)
        
        x= x + pos # (B, S, d_m)
        
        # pass through the transformer
        for block in self.transformer_blocks:
            x= block(x) # (B, S, d_m)
            
        x= self.final_norm(x) # (B, S, d_m)
        
        # pass through the final output layer
        x= self.final_output_projection(x) # (B, S, V)
        
        return x
    
    # weight initialisation
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
    
    
if __name__ == "__main__":
    x = torch.randint(1, 9, (2, GPT_CONFIG_124M["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    
    y = torch.randint(1, 9, (2, GPT_CONFIG_124M["context_length"]), dtype= torch.int32, device= DEVICE) # (B, S)
    
    
    model = GPT().to(DEVICE)
    
    logits = model(x)
    
    print("~ Input shape:", x.shape)
    print("~ Output shape:", y.shape)
    print("~ Logits shape:", logits.shape)
    


    # get the loss
    loss_func= nn.CrossEntropyLoss()
    
    ce_loss= loss_func(logits.to(torch.float32).flatten(0, 1), y.to(torch.long).view(-1))
    
    print(f"~ CE Loss: {ce_loss:.2f}")

    # check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n~ Total number of parameters: {total_params:,}")