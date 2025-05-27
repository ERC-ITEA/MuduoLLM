import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["STHeiti"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义维度和模型数据
categories = ["结构完整性", "内容准确性", "内容一致性", "语言逻辑性", "素养导向性"]
# categories = ["", "", "", "", ""]
models =  {
    "Qwen2.5-14B-instruct（阿里云）": [80.22, 54.13, 88.35, 87.30, 70.05],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [80.70, 64.31, 89.55, 89.70, 79.65],
    "Phi-4（微软）": [54.00, 34.24, 67.80, 72.75, 83.85],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [58.71, 32.06, 67.80, 60.75, 68.55],
    "Gemma3-12B（谷歌）": [80.37, 63.79, 89.40, 89.70, 81.90],
    "GLM-9B-chat（智谱）": [78.42, 50.01, 88.50, 84.15, 79.05],
    "Baichuan2-13B-Chat（百川）": [64.86, 35.57, 75.00, 77.25, 71.25],
    "EduChat-sft-002-13b（华东师范大学）": [47.58, 31.86, 31.20, 41.25, 58.50],
    "Confucius-o1（网易）": [81.75, 65.57, 89.70, 89.40, 76.65],
    "Spark-lite（讯飞）": [71.67, 37.71, 67.95, 66.30, 79.80],
    "师承万象（北京师范大学）": [83.76, 71.53, 88.20, 87.00, 84.75]
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
plt.savefig('../imgs/evals/radar_jiaoan.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()