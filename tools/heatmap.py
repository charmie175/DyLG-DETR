from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn as nn

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import warnings
warnings.filterwarnings('ignore')
import glob
import ast
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger
from ppdet.modeling.backbones.resnet import AffineLieGroupConv2D

logger = setup_logger('heatmap_infer')

# 热力图生成器类
class GradCAMHelper:
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.gradients = []
        self.forward_handle = None  # 明确区分前向/反向钩子
        self.backward_handles = []
        self.input_shape = None  # 新增属性
        
    def hook_fn(self, layer, input, output):
        self.activations.append(output.numpy())

    def grad_hook_fn(self, grad):
        self.gradients.insert(0, grad.numpy())
        return grad

    def find_target_layer(self):
        """针对该模型结构的层选择策略"""
        # 目标层特征
        target_type = AffineLieGroupConv2D
        candidate_patterns = [
            'res5.res5c.branch2b',  # 最深层
            'res4.res4f.branch2b',   # 中层高分辨率
            'branch2b2',             # 备选分支
            'branch2b'               # 通用分支
        ]
        
        # 遍历模型并记录候选层
        candidates = []
        for name, layer in self.model.named_sublayers():
            if isinstance(layer, target_type):
                # 计算匹配优先级
                priority = next((i for i, p in enumerate(candidate_patterns) if p in name), len(candidate_patterns))
                candidates.append( (priority, len(name), name, layer) )  # 优先级/名称长度/层信息
        
        # 选择最优层（优先级高者优先，同优先级选择路径较长者）
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            best_layer = candidates[0][3]
            print(f"最终选择：{candidates[0][2]}")
            return best_layer
        
        # 备选方案
        raise ValueError("模型中没有找到AffineLieGroupConv2D层，请检查模型结构")

    def generate(self, inputs_dict):
        try:
            self.input_shape = inputs_dict['image'].shape  
            # 清除历史梯度
            self.model.clear_gradients()
            
            # 前向传播
            output = self.model(inputs_dict)
            
            # 获取目标层
            target_layer = self.find_target_layer()
            
            # 注册前向钩子
            self.forward_handle = target_layer.register_forward_post_hook(
                lambda layer, input, output: self.activations.append(output.numpy())
            )
            
            # 重新前向传播以捕获激活
            output = self.model(inputs_dict)
            
            # 获取目标分数（确保在计算图中）
            bbox = output['bbox'].numpy()
            confidences = bbox[:, 4]
            target_class = confidences.argmax()
            
            # 创建可微分目标
            target_score = output['bbox'][target_class, 4]
            
            # 反向传播
            target_score.backward(retain_graph=True)
            
            # 获取梯度
            grads = []
            for param in target_layer.parameters():
                if param.grad is not None:
                    grads.append(param.grad.numpy())
            
            return self._compute_heatmap(grads)
            
        finally:
            if self.forward_handle:
                self.forward_handle.remove()

    def _compute_heatmap(self, grads):
        heatmap = None  # 初始化默认值
    
        valid_weights = []
        c_out = self.activations[0].shape[1]  # 激活图的输出通道数
    
        for g in grads:
            if len(g.shape) != 4:
                continue  # 跳过非卷积核梯度
        
        # 检查梯度输出通道是否匹配激活图
            if g.shape[0] == c_out:
            # 计算权重: [out_channels, in_channels] -> [out_channels]
                w = g.mean(axis=(1, 2, 3))  # 在in_channels和空间维度平均
                valid_weights.append(w)
    
        if not valid_weights:
            raise ValueError(f"未找到与激活图通道数({c_out})匹配的梯度")
    
    # 融合多个权重（平均）
        weights = np.stack(valid_weights).mean(axis=0)
    
    # 验证维度
        if weights.shape[0] != c_out:
            raise ValueError(f"权重维度({weights.shape})与激活通道数({c_out})不匹配")
    
    # 加权激活 [C, H, W]
        act = self.activations[0][0]
        heatmap = np.einsum('c,cwh->wh', weights, act)  # 精确矩阵乘法
    
    # 后处理
        heatmap = cv2.resize(heatmap, (self.input_shape[3], self.input_shape[2]))
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
        return heatmap

def parse_args():
    """扩展参数解析"""
    parser = ArgsParser()
    parser.add_argument("--infer_dir", type=str, default=None)
    parser.add_argument("--infer_img", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="heatmap_output")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="热力图叠加透明度")
    return parser.parse_args()

def preprocess(image_path, target_size=640, stride=32):
    """优化后的尺寸对齐预处理函数"""
    # 参数校验
    if not isinstance(image_path, (str, bytes, os.PathLike)):
        raise TypeError(f"无效路径类型: {type(image_path)}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法解码图像: {image_path}")
    orig_h, orig_w = img.shape[:2]

    max_size = max(orig_h, orig_w)
    scale = target_size / max_size
    new_h = int(round(orig_h * scale / stride)) * stride 
    new_w = int(round(orig_w * scale / stride)) * stride  

    # 二次校验
    assert new_h % stride == 0 and new_w % stride == 0, \
        f"尺寸对齐失败: {new_h}x{new_w} (stride={stride})"

    # 缩放图像（关键修改点）
    img_resized = cv2.resize(img, (new_w, new_h))  

    pad_h = max(target_size - new_h, 0)  # 防止负填充
    pad_w = max(target_size - new_w, 0)
    img_padded = cv2.copyMakeBorder(
        img_resized,
        0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )
    
    # 构建输入字典
    img_norm = img_padded.astype(np.float32) / 255.0
    inputs = {
        'image': paddle.to_tensor(img_norm.transpose(2,0,1)[None]),
        'im_shape': paddle.to_tensor([[orig_h, orig_w]], dtype='float32'),
        'scale_factor': paddle.to_tensor([[new_h/orig_h, new_w/orig_w]], dtype='float32')
    }
    
    return {
        'padded': img_padded,  # 填充后的图像 (640,640,3)
        'original': img,        # 原始未缩放图像 (1500,2000,3)
        'inputs': inputs,       # 模型输入
        'orig_hw': (orig_h, orig_w)  # 原始尺寸
    }


import os
import cv2

def get_test_images(infer_dir=None, infer_img=None):
    """安全获取测试图像路径列表"""
    # 参数有效性检查
    if not infer_dir and not infer_img:
        raise ValueError("至少需要指定一个输入路径")
    
    # 优先处理单张图像输入
    if infer_img:
        if not os.path.exists(infer_img):
            raise FileNotFoundError(f"指定图像不存在: {infer_img}")
        if os.path.isfile(infer_img):
            return [infer_img]
        elif os.path.isdir(infer_img):
            infer_dir = infer_img  # 兼容将目录误传为infer_img的情况
    
    # 处理目录输入
    if infer_dir:
        if not os.path.isdir(infer_dir):
            raise NotADirectoryError(f"输入路径不是目录: {infer_dir}")
        image_list = [
            os.path.join(infer_dir, f) 
            for f in os.listdir(infer_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))
        ]
        if not image_list:
            raise ValueError(f"目录中未找到有效图像: {infer_dir}")
        return image_list
    
    return []

def run(FLAGS, cfg):
    """主运行逻辑"""
    # 初始化模型
    trainer = Trainer(cfg, mode='test')
    trainer.model.eval()  # 强制进入评估模式
    trainer.model.training = False  # 明确关闭训练标志
    trainer.load_weights(cfg.weights)
    
    # 创建热力图生成器
    heat_generator = GradCAMHelper(trainer.model)
    
    # 获取输入图像
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    
    # 创建输出目录
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    for path in images:

        preproc_data = preprocess(path)
        orig_img = preproc_data['original']  # 直接获取原始未缩放图像 (1500,2000,3)
        inputs = preproc_data['inputs']
        
        heatmap = heat_generator.generate(inputs)  # 传递字典

        heatmap = cv2.resize(heatmap,(orig_img.shape[1], orig_img.shape[0]))  # 宽度在前，高度在后

        
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        heatmap_color = heatmap_color[:, :, :3]


        heatmap_float = heatmap_color.astype(np.float32) / 255
        orig_img_float = orig_img.astype(np.float32) / 255
        
        assert orig_img_float.shape == heatmap_float.shape, \
    f"尺寸校验失败: 原始图像{orig_img_float.shape} vs 热力图{heatmap_float.shape}"
        

        overlay = cv2.addWeighted(orig_img_float, 1-FLAGS.alpha, 
                         heatmap_float, FLAGS.alpha, 0)
        overlay = (overlay * 255).astype(np.uint8)

        # 保存结果
        save_path = os.path.join(
            FLAGS.output_dir, 
            os.path.basename(path))
        cv2.imwrite(save_path, overlay)
        logger.info(f"保存结果到: {save_path}")


def main():
    """主函数"""
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    
    # 设备设置
    place = 'gpu' if cfg.get('use_gpu', False) else 'cpu'
    paddle.set_device(place)
    
    # 配置检查
    check_config(cfg)
    check_gpu(cfg.get('use_gpu', False))
    check_version()
    
    run(FLAGS, cfg)

if __name__ == '__main__':
    main()