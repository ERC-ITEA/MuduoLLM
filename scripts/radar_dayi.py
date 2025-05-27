import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["STHeiti"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义维度和模型数据
categories = [
    "语言流畅", "知识点正确", "推理正确", "合理反馈", 
    "导正话题", "分步骤讲解", "提问质量", "引导质量"
]
# categories = ["", "", "", "", "", "", "", ""]

models =  {
    "Qwen2.5-14B-instruct（阿里云）": [82.99, 43.01, 86.93, 62.08, 76.38, 80.22, 84.39, 54.05],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [79.84, 44.02, 89.26, 75.04, 28.91, 15.85, 28.72, 50.84],
    "Phi-4（微软）": [74.54, 42.23, 77.00, 52.19, 56.09, 84.01, 89.03, 56.95],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [47.76, 41.65, 61.28, 30.38, 33.31, 29.93, 3.83, 8.09],
    "Gemma3-12B（谷歌）": [87.74, 32.85, 73.61, 64.77, 80.42, 77.09, 95.37, 70.32],
    "GLM-9B-chat（智谱）": [76.62, 44.52, 82.82, 65.49, 36.82, 66.76, 84.94, 68.56],
    "Baichuan2-13B-Chat（百川）": [71.14, 41.61, 74.05, 30.66, 15.17, 60.76, 23.84, 17.89],
    "EduChat-sft-002-13b（华东师范大学）": [87.90, 38.37, 71.55, 26.63, 38.49, 50.66, 55.15, 18.11],
    "Confucius-o1（网易）": [57.46, 40.70, 87.17, 64.82, 22.79, 15.30, 17.26, 50.61],
    "Spark-lite（讯飞）": [71.73, 39.08, 60.16, 28.13, 16.21, 28.99, 3.17, 23.24],
    "师承万象（北京师范大学）": [89.09, 82.26, 88.83, 77.57, 96.97, 90.28, 91.77, 74.34]
}

def create_radar_chart(ax, categories, models_data):
    """创建雷达图并解决文字重叠问题"""
    # 设置雷达图角度
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 设置坐标轴标签
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=16, color='black')
    
    # 设置网格线样式
    ax.grid(True, color='gray', linestyle='-', alpha=0.2)  # 设置网格线颜色和透明度
    ax.set_yticklabels([])  # 隐藏径向刻度值
    
    # 设置最外侧圆圈的颜色
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_alpha(0.2)
    
    # 调整标签位置，避免与雷达图冲突
    label_positions = ax.get_xticks()
    for i, label in enumerate(ax.get_xticklabels()):
        angle_rad = label_positions[i]
        # 将角度转换为0-360度范围
        angle_deg = np.degrees(angle_rad) % 360
        
        # 调整标签与雷达图的距离
        label.set_position((angle_rad, -0.1))  # 增大距离值以远离中心
        label.set_fontsize(20)  # 增大字体大小
        label.set_color('black')  # 设置标签颜色为黑色

    # 绘制每个模型的数据
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models_data)))
    lines = []
    for i, (model, scores) in enumerate(models_data.items()):
        # 确保数据闭合
        if len(scores) != num_vars:
            print(f"警告: {model} 的分数数量({len(scores)})与维度数量({num_vars})不匹配，跳过该模型")
            continue
            
        scores += scores[:1]
        line = ax.plot(angles, scores, linewidth=2, label=model, color=colors[i])
        ax.fill(angles, scores, alpha=0.2, color=colors[i])
        lines.extend(line)

    # 设置雷达图范围
    max_value = max([max(scores) for scores in models_data.values()])
    ax.set_ylim(0, max_value * 1.1)
    
    return lines

# 创建单个雷达图
fig = plt.figure(figsize=(20, 20))  # 增大图形大小
ax = fig.add_subplot(111, polar=True)

# 绘制雷达图
lines = create_radar_chart(ax, categories, models)

# 添加图例
legend = ax.legend(lines, models.keys(), 
                  loc='center left',
                  bbox_to_anchor=(1.2, 0.8),
                  fontsize=18,
                  frameon=False)

# 调整布局 
plt.tight_layout()

# 保存图片
plt.savefig('../imgs/evals/radar_dayi.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()    