import torch
import torch.nn as nn
from transformers import BertModel, PretrainedBartModel
from typing import Union, Optional
from itertools import chain

class DyT(nn.Module):
    """Dynamic Tanh Transformation module."""
    def __init__(self, n_num_features: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.weight = nn.Parameter(torch.ones(n_num_features))
        self.bias = nn.Parameter(torch.zeros(n_num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class Label(nn.Module):
    def __init__(
        self, 
        bert, 
        tokenizer, 
        label_list: torch.Tensor,
        hidden_dim: int = 768,
        num_heads: int = 8,
        max_length: int = 128, 
        ):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_labels = label_list.shape[0]
        self.labels = nn.Parameter(label_list)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.layerNorm = DyT(hidden_dim)
        
    def forward(self, prompts: list[list]| None):
        if self.training:
            if prompts is None:
                raise ValueError("Prompts required during training")
            all_sentences = list(chain.from_iterable(prompts)) # -> list
            
            with torch.no_grad():
                inputs = self.tokenizer(
                        all_sentences,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                ).to(self.labels.device)
                outputs = self.bert(**inputs)
                prompts_hidden = outputs.last_hidden_state[:, 0, :].to(self.labels.device) # -> [sentences_num, hidden_dim]
            
            grouped_prompts = prompts_hidden.view(self.n_labels, -1, self.labels.shape[-1])
            queries = self.labels.unsqueeze(1)
            attn_output, _ = self.attn(queries, grouped_prompts, grouped_prompts)
            updated_labels = attn_output.squeeze(1)
            
            with torch.no_grad():
                self.labels.data.copy_(updated_labels.data)
            labels_features = self.layerNorm(self.labels)
            return labels_features
        else:
            if prompts is not None:
                raise RuntimeError("Prompts should not be provided during evaluation")
            labels_features = self.layerNorm(self.labels).detach()
            return labels_features
        
class ProjectMLP(nn.Module):
    def __init__(
        self, *,
        n_labels: int,
        hidden_dim: int = 768,
        project_dim: int = 512,
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.project_dim = project_dim
        
        self.R = nn.Parameter(torch.empty(n_labels, hidden_dim))
        self.W = nn.Parameter(torch.empty(project_dim, hidden_dim))
        self.S = nn.Parameter(torch.empty(n_labels, project_dim))
        self.bias = nn.Parameter(torch.empty(n_labels, project_dim))
        
        self.reset_parameters()
        
    def forward(self, x: torch.Tensor):
        # x [Batch_size, n_labels, hidden_dim]
        x = x * self.R
        x = x @ self.W.T
        x = x * self.S
        
        x = x + self.bias
        return x
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.ones_(self.R)
        nn.init.ones_(self.S)
    
        bound = 1 / (self.bias.size(1) ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

class QDTL(nn.Module):
    def __init__(
        self,
        bert_model,
        tokenizer,
        label_list: torch.Tensor,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        n_deep:int =  3,
        d_wide: int = 512
        ):
        super().__init__()
        self.encoder = self._init_bert(bert_model)
        self.label_model = Label(bert=bert_model, tokenizer=tokenizer, hidden_dim=hidden_dim, label_list=label_list)
        self.cross_Attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.blocks = nn.ModuleList()
        for i in range(n_deep):
            block = []
            block.append(
                ProjectMLP(
                    n_labels = label_list.shape[0],
                    hidden_dim = hidden_dim if i==0 else d_wide, 
                    project_dim = d_wide
                )
            )
            block.append(nn.ReLU())
            block.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block))
            
        self.output = nn.Linear(d_wide, 3)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: list[list] = None 
    ) -> torch.Tensor:
        # get labels_features[n_labels, hidden_dim]
        labels_features = self.label_model(prompts)
        
        # get text features [batch_size, seq_len, hidden_dim]
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        text_features = encoder_outputs.last_hidden_state
        
        # CrossAttn
        labels_features = labels_features.unsqueeze(0).repeat(text_features.shape[0], 1, 1)
        cross_features, _ =self.cross_Attn(
            query=labels_features,
            key=text_features,
            value=text_features
        )# [Batch_size, n_labels, hidden_dim]
        
        # MLP project
        for i, block in enumerate(self.blocks):
            cross_features = block(cross_features)
        
        output = self.output(cross_features)
        return output

    def _init_bert(
        self, 
        bert: Optional[Union[str, PretrainedBartModel, BertModel]]
    ) -> BertModel:
        if bert is None:
            return BertModel.from_pretrained("bert-base-uncased")
        elif isinstance(bert, str):
            return BertModel.from_pretrained(bert)
        elif isinstance(bert, (BertModel, PretrainedBartModel)):
            return bert
        raise TypeError("Invalid bert type. Expected str or PretrainedBartModel")