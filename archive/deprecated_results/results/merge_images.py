#!/usr/bin/env python3
import os
import re
from PIL import Image
import glob

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

def get_tau_sort_key(tau):
    """获取tau值的排序键，inf排在最后"""
    if tau == float('inf'):
        return (1, 0)  # 第二个值确保inf只有一个
    else:
        return (0, tau)

def merge_images_by_dim():
    """按dim分组合并图片"""
    # 获取所有PNG文件
    png_files = glob.glob('dim_*.png')
    
    # 按dim分组
    dim_groups = {}
    for filepath in png_files:
        filename = os.path.basename(filepath)
        dim, tau = extract_dim_tau(filename)
        if dim is not None:
            if dim not in dim_groups:
                dim_groups[dim] = []
            dim_groups[dim].append((tau, filepath))
    
    # 处理每个dim组
    for dim in sorted(dim_groups.keys()):
        print(f"处理 dim={dim} 组...")
        
        # 按tau排序
        images_data = dim_groups[dim]
        images_data.sort(key=lambda x: get_tau_sort_key(x[0]))
        
        # 加载图片
        images = []
        for tau, filepath in images_data:
            try:
                img = Image.open(filepath)
                images.append(img)
                tau_str = 'inf' if tau == float('inf') else str(tau)
                print(f"  加载: tau={tau_str}")
            except Exception as e:
                print(f"  错误: 无法加载 {filepath}: {e}")
        
        if not images:
            print(f"  警告: dim={dim} 组没有有效图片")
            continue
        
        # 合并图片（3x3网格排列）
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        max_height = max(heights)
        
        # 创建3x3网格图片
        grid_cols = 3
        grid_rows = 3
        total_width = max_width * grid_cols
        total_height = max_height * grid_rows
        
        merged_image = Image.new('RGB', (total_width, total_height))
        
        # 粘贴图片到网格位置
        for i, img in enumerate(images):
            row = i // grid_cols
            col = i % grid_cols
            x_offset = col * max_width
            y_offset = row * max_height
            merged_image.paste(img, (x_offset, y_offset))
        
        # 保存合并后的图片
        output_filename = f'dim_{dim}.png'
        merged_image.save(output_filename)
        print(f"  保存: {output_filename} (尺寸: {merged_image.size})")
        
        # 关闭图片
        for img in images:
            img.close()
        merged_image.close()
    
    print(f"\n完成！共处理了 {len(dim_groups)} 个dim组")

if __name__ == "__main__":
    merge_images_by_dim()
