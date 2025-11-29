#!/usr/bin/env python3
import os
import re
from PIL import Image
import math

def extract_dim_tau(filename):
    """从文件名中提取dim和tau值"""
    match = re.match(r'dim_(\d+)_tau_(.+)\.png$', filename)
    if match:
        dim = int(match.group(1))
        tau_str = match.group(2)
        # 处理inf的情况
        if tau_str == 'inf':
            tau = float('inf')
        else:
            tau = int(tau_str)
        return dim, tau
    return None, None

def get_tau_order(tau):
    """定义tau的排序顺序"""
    if tau == float('inf'):
        return 1000000  # 把inf放在最后
    return tau

def main():
    # 获取当前目录中的所有PNG文件
    current_dir = '.'
    png_files = [f for f in os.listdir(current_dir) if f.endswith('.png') and 'dim_' in f and 'tau_' in f]
    
    # 按dim分组
    dim_groups = {}
    for filename in png_files:
        dim, tau = extract_dim_tau(filename)
        if dim is not None and tau is not None:
            if dim not in dim_groups:
                dim_groups[dim] = []
            dim_groups[dim].append((tau, filename))
    
    # 对每组按tau排序
    for dim in dim_groups:
        dim_groups[dim].sort(key=lambda x: get_tau_order(x[0]))
    
    # 处理每个dim组
    for dim in sorted(dim_groups.keys()):
        images_info = dim_groups[dim]
        print(f"处理 dim={dim}, 共{len(images_info)}张图片")
        
        # 加载所有图片
        images = []
        for tau, filename in images_info:
            try:
                img = Image.open(filename)
                images.append((tau, img))
                print(f"  加载: {filename}")
            except Exception as e:
                print(f"  错误: 无法加载 {filename}: {e}")
        
        if not images:
            continue
        
        # 计算网格布局 (尽量接近正方形)
        num_images = len(images)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        
        # 获取第一张图片的尺寸作为参考
        first_img = images[0][1]
        img_width, img_height = first_img.size
        
        # 创建合并后的大图
        merged_width = cols * img_width
        merged_height = rows * img_height
        merged_image = Image.new('RGB', (merged_width, merged_height), color='white')
        
        # 将小图粘贴到大图上
        for i, (tau, img) in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            merged_image.paste(img, (x, y))
            print(f"  粘贴到位置 ({col}, {row}): tau={tau}")
        
        # 保存合并后的图片
        output_filename = f"dim_{dim}_merged.png"
        merged_image.save(output_filename)
        print(f"  保存: {output_filename}")
        print(f"  布局: {rows}行 x {cols}列")
        print()

if __name__ == "__main__":
    main()
