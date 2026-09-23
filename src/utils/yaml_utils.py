import yaml

# def load_yaml(domain_config_path: str):
#     # 1. Đọc config chung
#     with open("configs/base.yaml", "r", encoding="utf-8") as f:
#         base_config = yaml.safe_load(f)
        
#     # 2. Đọc config riêng của domain (VD: exam_cheating.yaml)
#     with open(domain_config_path, "r", encoding="utf-8") as f:
#         domain_config = yaml.safe_load(f)
        
#     # 3. Gộp config (Domain config sẽ ghi đè các tham số trùng trong base)
#     merged_config = {**base_config, **domain_config}
#     return merged_config

def load_yaml(domain_config_path: str):
    with open(domain_config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
        
    return yaml_data