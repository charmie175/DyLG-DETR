import json
import copy

def fix_coco_annotations(input_path, output_path):
    # 加载原始标注
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 创建修复版本
    fixed_data = copy.deepcopy(data)
    
    for ann in fixed_data['annotations']:
        seg = ann['segmentation']
        bbox = ann['bbox']
        
        # 如果分割是空列表，使用边界框创建矩形分割
        if isinstance(seg, list) and len(seg) == 0:
            x, y, w, h = bbox
            # 创建矩形多边形 [x1, y1, x2, y2, x3, y3, x4, y4]
            ann['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    
    # 保存修复后的标注
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f)
    
    print(f"修复后的标注已保存到: {output_path}")
    return fixed_data

# 修复标注文件
fixed_annotation_path = '/root/autodl-tmp/visdrone/val_fixed.json'
fixed_data = fix_coco_annotations('/root/autodl-tmp/visdrone/val.json', fixed_annotation_path)

# 验证修复结果
print(f"修复前分割空列表数量: 38759")
print(f"修复后分割空列表数量: {sum(1 for ann in fixed_data['annotations'] if isinstance(ann['segmentation'], list) and len(ann['segmentation']) == 0)}")