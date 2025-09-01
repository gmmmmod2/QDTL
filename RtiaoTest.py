from train import train
import json
import yaml

def TrainRatios(model_name, model_config, experiment_config):
    train_ratios = [round(i * 0.1, 1) for i in range(1, 7)]
    for i , train_ratio in enumerate(train_ratios):
        remaining_ratio = 1.0 - train_ratio
        val_test_ratio = round(remaining_ratio / 2, 2)
        
        datapath_config = "configs/datatest.json"
        with open(datapath_config, 'r', encoding='utf-8') as f:
            current_config = json.load(f)
        
        current_config['Datasets 1']["train_ratio"] = train_ratio
        current_config['Datasets 1']["val_ratio"] = val_test_ratio
        current_config['Datasets 1']["test_ratio"] = val_test_ratio
        
        print(f"正在使用训练集比例 {train_ratio} 进行实验 (验证集: {val_test_ratio}, 测试集: {val_test_ratio})")
        with open(datapath_config, 'w', encoding='utf-8') as f:
            json.dump(current_config, f, indent=4, ensure_ascii=False)
        
        test_name = f"{model_name}_{i}"
        train(model_name, model_config, experiment_config, datapath_config, test_name=test_name)

def RandomSeed(model_name, model_config, experiment_config, datapath_config):
    random_seed = [i for i in range(0,5)]
    
    for i in random_seed:
        with open(experiment_config, 'r', encoding='utf-8') as f:
            current_config =  yaml.safe_load(f)
        current_config['experiment']['seed'] = i
        
        print(f"正在使用随机数种子 {i} ")
        
        with open(experiment_config, 'w', encoding='utf-8') as f:
            yaml.dump(current_config, f)
        test_name = f"{model_name}_{i}"
        
        train(model_name, model_config, experiment_config, datapath_config, test_name=test_name)
    
if __name__ == '__main__':
    model_name = 'QTRL'
    model_config = f"configs/{model_name}.toml"
    experiment_config = "configs/experiment.yaml"
    datapath_config = "configs/paths.json"
    # RandomSeed(model_name, model_config, experiment_config, datapath_config)
    TrainRatios(model_name, model_config, experiment_config)