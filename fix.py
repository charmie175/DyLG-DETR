import json
import os

def filter_invalid_bboxes(annotation_file):
    """过滤掉面积为零的无效边界框"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    valid_annotations = []
    removed_count = 0
    
    for ann in data['annotations']:
        # 检查边界框是否有效
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        
        # 计算面积
        area = w * h
        
        # 如果宽度或高度为0，或者面积为0，则跳过
        if w <= 0 or h <= 0 or area <= 0:
            print(f"移除无效边界框: 图像ID {ann['image_id']}, 坐标 {bbox}, 面积 {area}")
            removed_count += 1
            continue
        
        # 确保坐标是整数
        ann['bbox'] = [float(x1), float(y1), float(w), float(h)]
        valid_annotations.append(ann)
    
    # 更新数据
    data['annotations'] = valid_annotations
    
    # 保存修复后的文件
    output_file = annotation_file.replace('.json', '_filtered.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"原始标注数量: {len(data['annotations']) + removed_count}")
    print(f"有效标注数量: {len(data['annotations'])}")
    print(f"移除的无效标注: {removed_count}")
    print(f"保存到: {output_file}")
    
    return output_file

# 修复训练和验证标注文件
train_filtered = filter_invalid_bboxes('/root/autodl-tmp/DIOR_json/Annotations/train.json')
val_filtered = filter_invalid_bboxes('/root/autodl-tmp/DIOR_json/Annotations/val.json')