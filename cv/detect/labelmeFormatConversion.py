import os
import argparse
import json  

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Convert JSON files to TXT with identical train/val sets')
    parser.add_argument('--json_path', default='C:\\cv\\jsonfile', type=str, 
                        help='Root directory containing JSON files (including subfolders)')
    parser.add_argument('--output_root', default='dataset', type=str, 
                        help='Root directory where train/val sets will be created')
    
    opt = parser.parse_args()  # 移除val_ratio参数，因为不需要划分比例

    # 检查 JSON 文件根目录是否存在
    if not os.path.exists(opt.json_path):
        print(f"Error: JSON root directory not found - {opt.json_path}")
        print("Please check the path or use --json_path to specify a different directory")
        return

    # 确保输出根目录存在
    os.makedirs(opt.output_root, exist_ok=True)

    # 收集所有 JSON 文件路径
    all_json_files = []
    # 递归获取所有 JSON 文件（包括子文件夹）
    for root, dirs, files in os.walk(opt.json_path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                all_json_files.append(json_path)
    
    if not all_json_files:
        print(f"Error: No JSON files found in {opt.json_path} or its subfolders")
        return

    # 统计总文件数
    total_files = len(all_json_files)
    print(f"Found {total_files} JSON files in {opt.json_path} (including subfolders)")
    print(f"Will convert to TXT and create identical train/val sets with {total_files} files each")

    # 创建训练集和验证集根目录
    train_root = os.path.join(opt.output_root, 'train')
    val_root = os.path.join(opt.output_root, 'val')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # 转换所有 JSON 文件为 TXT，并同时复制到 train 和 val 文件夹
    for idx, json_path in enumerate(all_json_files):
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(json_path))[0]
        # 构建 train 和 val 文件夹中的 TXT 文件路径
        txt_path_train = os.path.join(train_root, f"{file_name}.txt")
        txt_path_val = os.path.join(val_root, f"{file_name}.txt")
        
        try:
            # 读取 JSON 文件内容
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 同时写入 train 文件夹
            with open(txt_path_train, 'w', encoding='utf-8') as f_train:
                f_train.write(str(json_data))
            print(f"Converted to train TXT: {txt_path_train}")

            # 同时写入 val 文件夹（内容完全相同）
            with open(txt_path_val, 'w', encoding='utf-8') as f_val:
                f_val.write(str(json_data))
            print(f"Converted to val TXT: {txt_path_val}")

        except Exception as e:
            print(f"Error converting {json_path} to TXT: {str(e)}")

    print(f"\nAll JSON files have been converted to TXT in {opt.output_root}")
    print(f"Train set: {train_root} with {len(all_json_files)} files")
    print(f"Validation set: {val_root} with {len(all_json_files)} files")
    print("Train and val sets are identical in quantity, content, and order.")

if __name__ == "__main__":
    main()