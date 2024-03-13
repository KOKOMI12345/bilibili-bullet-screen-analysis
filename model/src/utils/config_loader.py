import yaml

def load_global_config(file_path='/mnt/workspace/bilibili-bullet-screen-analysis/model/config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    global_config = config.get('global_config', {})
    return global_config