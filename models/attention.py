import torch 
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, number_of_examples=100, embed_dim=784, number_of_heads=4):
        # Model init
        super(Attention, self).__init__()

        # MLP 
        self.img_mlp = nn.Sequential(
            nn.Linear(784, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Multi head attention 
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=number_of_heads,
            batch_first=True
        )
    
    def forward(self, img, values):
        # Run image and values through MLP
        img_ = self.img_mlp(img)
        values_ = self.img_mlp(values)

        # Forward attention 
        output_attn, output_weights_attn = self.multi_head_attention(img_, values_, values_)

        # Do a weighted sum between the attention output and the pixel values
        output = torch.bmm(output_weights_attn, values)
        return output, output_weights_attn


