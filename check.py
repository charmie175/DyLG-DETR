import json

def check_annotations(ann_file):
    """检查标注文件中的类别ID"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"标注文件: {ann_file}")
    print(f"图像数量: {len(data['images'])}")
    print(f"标注数量: {len(data['annotations'])}")
    print(f"类别数量: {len(data['categories'])}")
    
    # 检查类别信息
    print("\n类别信息:")
    for cat in data['categories']:
        print(f"  ID: {cat['id']}, 名称: {cat['name']}")
    
    # 检查标注中的类别ID范围
    cat_ids_in_anns = set()
    for ann in data['annotations']:
        cat_ids_in_anns.add(ann['category_id'])
    
    print(f"\n标注中出现的类别ID: {sorted(cat_ids_in_anns)}")
    print(f"最大类别ID: {max(cat_ids_in_anns)}")
    
    # 检查是否有超出预期范围的类别ID
    expected_max_id = len(data['categories']) - 1  # 如果从0开始
    if max(cat_ids_in_anns) > expected_max_id:
        print(f"警告: 发现超出预期的类别ID! 预期最大ID: {expected_max_id}, 实际最大ID: {max(cat_ids_in_anns)}")
    
    return data

# 检查训练集和验证集标注
train_ann = '/root/autodl-tmp/DIOR_json/Annotations/train.json'
val_ann = '/root/autodl-tmp/DIOR_json/Annotations/val.json'

train_data = check_annotations(train_ann)
print("\n" + "="*50 + "\n")
val_data = check_annotations(val_ann)

def analyze_predictions(pred_file):
    """分析预测结果中的类别ID"""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"预测结果总数: {len(predictions)}")
    
    # 统计类别ID分布
    cat_id_counts = {}
    for pred in predictions:
        cat_id = pred['category_id']
        cat_id_counts[cat_id] = cat_id_counts.get(cat_id, 0) + 1
    
    print("\n预测结果中的类别ID分布:")
    for cat_id in sorted(cat_id_counts.keys()):
        print(f"  类别ID {cat_id}: {cat_id_counts[cat_id]} 个预测")
    
    # 检查是否有超出20的类别ID
    invalid_cat_ids = [cat_id for cat_id in cat_id_counts.keys() if cat_id >= 20]
    if invalid_cat_ids:
        print(f"\n警告: 发现无效的类别ID: {invalid_cat_ids}")
        print(f"这些预测占总预测的: {sum(cat_id_counts[cat_id] for cat_id in invalid_cat_ids) / len(predictions) * 100:.2f}%")
    
    return cat_id_counts

# 分析预测结果
pred_file = '/root/runs/detect/val/predictions.json'
cat_id_counts = analyze_predictions(pred_file)