import torch
import os
import json
from transformers import BertModel, BertTokenizer
from utils.other import get_model
from data.DataLoader import DataLoaderModule
from training.metrics import calculate_metricsALL

def evaluate(model_name, path, model_path, config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config['model']['tokenizer'])
    
    data_config = {
        'csv_path': config['data']['Datasets 1']['path'],
        'max_length': config['model']['max_length'],
        'batch_size': config['experiment']['training']['batch_size'],
        'train_ratio': config['data']['Datasets 1']['train_ratio'],
        'val_ratio': config['data']['Datasets 1']['val_ratio'],
        'test_ratio': config['data']['Datasets 1']['test_ratio'],
        'random_seed': config['experiment']['experiment']['seed'],
        'num_workers': min(4, os.cpu_count() - 1),
    }
    data_module = DataLoaderModule(data_config, tokenizer)
    _, _, test_loader = data_module.get_dataloaders()
    
    bert = BertModel.from_pretrained(config['model']['bert_model']).to(device)
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()
    
    model = get_model(model_name, bert, tokenizer, device, config['model'])
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
            targets = batch['target']
            
            outputs = model(**inputs)
            all_outputs.append(outputs)
            all_targets.extend(targets)
    
    metrics = calculate_metricsALL(all_outputs, all_targets)
    print(metrics)
    metrics.to_csv(f'{path}.csv')

if __name__ == "__main__":
    model_name = 'QTRL'
    path = 'QTRL_0'
    model_path = f"experiments/{path}/best_model.pth"
    config_path = f"experiments/{path}/config.json"
    evaluate(model_name, path, model_path, config_path)
