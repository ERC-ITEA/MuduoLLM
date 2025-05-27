import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["STHeiti"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义维度和模型数据
subjects = ["数学", "化学", "物理"]

categories = {"数学": ['图形与几何', '数与代数', '统计与概率', '综合与实践'],
              "化学": ["化学与社会·\n跨学科实践", "物质的化学变化", "物质的性质\n与应用", "物质的组成\n与结构", "科学探究与\n化学实验"],
              "物理": ["实验探究", "物质", "能量", "跨学科实践", "运动和相互作用"]
}

# categories = {"数学": ['', '', '', ''],
#               "化学": ["", "", "", "", ""],
#               "物理": ["", "", "", "", ""]
# }

models = {"化学": {
    "Qwen2.5-14B-instruct（阿里云）": [76.00, 68.00, 77.25, 73.25, 59.75],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [69.00, 69.75, 74.75, 70.50, 59.00],
    "Phi-4（微软）": [58.25, 52.00, 59.00, 53.00, 34.75],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [46.00, 28.00, 45.75, 35.75, 25.25],
    "Gemma3-12B（谷歌）": [63.25, 51.75, 63.75, 59.25, 39.75],
    "GLM-9B-chat（智谱）": [69.75, 62.00, 71.75, 67.25, 55.50],
    "Baichuan2-13B-Chat（百川）": [30.75, 26.75, 36.00, 28.75, 19.25],
    "EduChat-sft-002-13b（华东师范大学）": [33.75, 14.50, 29.50, 21.75, 17.00],
    "Confucius-o1（网易）": [75.75, 69.25, 77.25, 73.50, 62.50],
    "Spark-lite（讯飞）": [38.50, 22.75, 40.50, 39.50, 19.75],
    "师承万象（北京师范大学）": [91.50, 84.00, 91.50, 91.00, 81.00]
},
"数学": {
    "Qwen2.5-14B-instruct（阿里云）": [79.25, 79.50, 79.00, 71.50],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [80.25, 85.00, 78.75, 74.00],
    "Phi-4（微软）": [69.25, 74.00, 75.25, 59.25],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [44.00, 44.50, 43.00, 32.50],
    "Gemma3-12B（谷歌）": [73.25, 75.00, 74.25, 61.75],
    "GLM-9B-chat（智谱）": [57.00, 61.75, 58.25, 48.50],
    "Baichuan2-13B-Chat（百川）": [33.75, 41.75, 35.25, 31.00],
    "EduChat-sft-002-13b（华东师范大学）": [15.75, 18.00, 19.50, 16.50],
    "Confucius-o1（网易）": [78.25, 82.25, 75.00, 68.75],
    "Spark-lite（讯飞）": [24.00, 27.75, 27.00, 20.50],
    "师承万象（北京师范大学）": [87.00, 90.50, 88.50, 81.00]
},
"物理": {
    "Qwen2.5-14B-instruct（阿里云）": [66.25, 82.75, 83.25, 83.50, 83.75],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [60.75, 79.50, 80.00, 79.50, 78.75],
    "Phi-4（微软）": [44.75, 66.25, 71.00, 64.25, 65.50],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [32.75, 48.00, 45.50, 48.75, 49.50],
    "Gemma3-12B（谷歌）": [51.75, 71.75, 73.50, 71.75, 69.00],
    "GLM-9B-chat（智谱）": [54.00, 76.25, 74.00, 76.25, 75.00],
    "Baichuan2-13B-Chat（百川）": [24.75, 40.50, 42.50, 41.25, 45.75],
    "EduChat-sft-002-13b（华东师范大学）": [22.50, 25.25, 25.50, 31.00, 28.25],
    "Confucius-o1（网易）": [68.25, 78.50, 83.75, 80.50, 83.25],
    "Spark-lite（讯飞）": [23.75, 37.50, 33.50, 41.75, 41.25],
    "师承万象（北京师范大学）": [78.00, 94.50, 93.00, 93.50, 92.50]
}
}

def create_radar_chart(ax, subject, categories, models_data):
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
        label.set_position((angle_rad, -0.1))  # 增大距离值以远离中心, -0.1
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
    
    # 添加标题（放在图片下方）
    ax.set_title(subject, size=30, pad=20, y=-0.4)  # 使用y参数将标题位置调整到下方, -0.4
    
    return lines

# 创建一个大图包含三个子图和一个图例区域
fig = plt.figure(figsize=(30, 10))  # 增大整体图形大小
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.25])  # 稍微减小图例区域的宽度

# 创建三个雷达图
axes = []
for i in range(3):
    ax = fig.add_subplot(gs[0, i], polar=True)
    axes.append(ax)

# 绘制每个学科的雷达图
lines = []
for i, subject in enumerate(subjects):
    lines.extend(create_radar_chart(axes[i], subject, categories[subject], models[subject]))

# 添加共享图例（只使用第一个子图的线条）
unique_lines = []
unique_labels = []
seen_labels = set()
for line in lines:
    label = line.get_label()
    if label not in seen_labels:
        unique_lines.append(line)
        unique_labels.append(label)
        seen_labels.add(label)

# 创建图例区域
legend_ax = fig.add_subplot(gs[0, 3])
legend_ax.axis('off')  # 隐藏坐标轴
legend = legend_ax.legend(unique_lines, unique_labels, 
                         loc='center left',
                         fontsize=18,  # 减小图例文字大小
                         frameon=False)  # 移除图例边框

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('../imgs/evals/radar_zhuanye.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()    