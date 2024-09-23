from huggingface_hub import HfApi, HfFolder
import json
from datasets import Dataset, DatasetDict
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]  # 假设 JSON 文件是每行一个 JSON 对象的 JSON Lines 格式
    return data

# 定义文件路径
isabelle_train = load_json_file('D:\\桌面\\MMA-dataset\\isabelle_train.jsonl')
isabelle_val = load_json_file('D:\\桌面\\MMA-dataset\\isabelle_val.jsonl')
lean_test = load_json_file('D:\\桌面\\MMA-dataset\\lean_test.jsonl')
lean_train = load_json_file('D:\\桌面\\MMA-dataset\\lean_train.jsonl')
lean_val = load_json_file('D:\\桌面\\MMA-dataset\\lean_val.jsonl')

# 使用 datasets.from_list 创建数据集
isabelle_train_dataset = Dataset.from_list(isabelle_train)
isabelle_val_dataset = Dataset.from_list(isabelle_val)
lean_test_dataset = Dataset.from_list(lean_test)
lean_train_dataset = Dataset.from_list(lean_train)
lean_val_dataset = Dataset.from_list(lean_val)

# 将数据集组合成一个 DatasetDict
dataset_dict = DatasetDict({
    'isabelle_train': isabelle_train_dataset,
    'isabelle_val': isabelle_val_dataset,
    'lean_test': lean_test_dataset,
    'lean_train': lean_train_dataset,
    'lean_val': lean_val_dataset,
})

# 保存 Hugging Face token
hf_token = "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"
HfFolder.save_token(hf_token)


# 上传数据集到 Hugging Face Hub
repo_name = "AI-MO/MMA-dataset"
dataset_dict.push_to_hub(repo_name, token=hf_token,private=True)
