import json

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = []
    for item in data:
        data.append(item)
        
    return data

def save_json(json_file, data):

    with open(f"./{json_file}", "w") as file:
        json.dump(data, file)