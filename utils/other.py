import torch
from typing import List
from datetime import datetime
from transformers import BertModel, PretrainedBartModel
from typing import Any, Dict, Type, Optional, Union

from models.baseline import BMLP
from models.compared import BAspect, BGLU, BLSTM, BMultiConv
from models.new import QTRL
from models.ResNet import ResNet

ModelType = Type[torch.nn.Module]
MODELS_LIST: Dict[str, ModelType] = {
    'BMLP': BMLP,
    'QTRL': QTRL,
    'ResNet': ResNet,
    'BAspect': BAspect,
    'BGLU': BGLU,
    'BLSTM': BLSTM,
    'BMultiConv': BMultiConv
}
        
def init_bert(bert: Optional[Union[str, PretrainedBartModel, BertModel]]) -> BertModel:
    if bert is None:
        return BertModel.from_pretrained("bert-base-uncased")
    elif isinstance(bert, str):
        return BertModel.from_pretrained(bert)
    elif isinstance(bert, (BertModel, PretrainedBartModel)):
        return bert
    raise TypeError("Invalid bert type. Expected str or PretrainedBartModel")

def getEmbeddingsFlabels(label_list: List, bert, tokenizer, device, max_length: int = 128):
    bert.eval()
    embeddings = []
    with torch.no_grad():
        for label in label_list:
            inputs = tokenizer(label, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            outputs = bert(**inputs)
            # [CLS]
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)
            
    return torch.stack(embeddings).squeeze(1).to(device=device)   # [n_labels, hidden_size]

def getPromptEmbeddings(prompts_2d:List[List], bert, tokenizer, device, batch_size=32, max_length: int = 128):
    all_sentences = []
    label_indices = []
    for label_idx, sentences in enumerate(prompts_2d):
        all_sentences.extend(sentences)
        label_indices.extend([label_idx] * len(sentences))
    
    embeddings = torch.zeros(len(all_sentences), 768, device=device)
    
    bert.eval()
    with torch.no_grad():
        for i in range(0, len(all_sentences), batch_size):
            batch_sentences = all_sentences[i:i+batch_size]
            
            # Tokenize并移动到设备
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # 获取模型输出
            outputs = bert(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1)
            embeddings[i:i+batch_size] = mean_pooled
            
    batch_embeddings = torch.zeros(5, 768, device=device)
    for label in range(5):
        mask = torch.tensor(label_indices, device=device) == label
        batch_embeddings[label] = embeddings[mask].mean(dim=0)
        
    return batch_embeddings

def saveOutput(outputs, targets):
    log_file = "output_log.txt"
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    outputs_str = "[" + ", ".join([f"{x:.4e}" for x in outputs[:50]]) + ", ...]" 
    targets_str = "[" + ", ".join([f"{x:.4f}" for x in targets[:50]]) + ", ...]"
    
    with open(log_file, "a") as f:
        f.write(f"{current_time}:\n")
        f.write(f"    outputs: {outputs_str}\n")
        f.write(f"    targets: {targets_str}\n\n")

def calculate_baseline_metrics(df, columns=['Professionalism_score', 'Occupational_score', 'Effectiveness_score', 'Quality_score', 'Other_score']):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        results = {}
        all_y_true = []
        all_y_pred = []
        
        for col in columns:
            # 计算当前列的平均值
            mean_val = df[col].mean()
            
            # 创建全为平均值的预测数组
            y_true = df[col].values
            y_pred = np.full_like(y_true, mean_val)
            
            # 收集所有列的数据用于总体计算
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # 计算各项指标
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results[col] = {
                'mean_value': mean_val,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        
        # 计算总体指标
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        overall_r2 = r2_score(all_y_true, all_y_pred)
        
        # 添加总体结果
        results['Overall'] = {
            'mean_value': np.nan,  # 总体没有单一的平均值
            'RMSE': overall_rmse,
            'MAE': overall_mae,
            'R2': overall_r2
        }
        
        return pd.DataFrame.from_dict(results, orient='index')

def get_model(model_name: str, bert, tokenizer, device: torch.device, params: Dict[str, Any]) -> torch.nn.Module:
    if model_name not in MODELS_LIST:
        valid_names = ", ".join(MODELS_LIST.keys())
        raise ValueError(f"Invalid model name '{model_name}'. Valid options: {valid_names}")
    model_class = MODELS_LIST[model_name]
    
    model_params = params.copy()
    model_params.pop('bert_model', None)
    model_params.pop('tokenizer', None)
    max_length = model_params.pop('max_length', 128)
    
    label_embeddings = None
    if 'label_list' in model_params:
        label_list = model_params.pop('label_list')  
        label_embeddings = getEmbeddingsFlabels(label_list, bert, tokenizer, device, max_length)
    
    init_args = {
        'bert_model': bert,
        **model_params
    }
    if label_embeddings is not None:
        init_args['label_list'] = label_embeddings
    
    if model_name is 'QTRL':
        model = QTRL(
            bert_model=bert,
            tokenizer=tokenizer,
            label_list=label_embeddings,
            hidden_dim=768,
            num_heads=12,
            dropout=0.1
        ).to(device)
        return model
    
    try:
        model = model_class(**init_args).to(device)
        return model
    except TypeError as e:
        missing = set(model_class.__init__.__code__.co_varnames) - set(init_args.keys())
        raise RuntimeError(
            f"Model initialization failed: {str(e)}\n"
            f"Required params: {list(missing)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize {model_name}: {str(e)}"
        ) from e