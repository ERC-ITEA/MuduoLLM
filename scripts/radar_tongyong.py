import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["STHeiti"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义维度和模型数据
categories = ["IFEval", "CEval", "MMLU", "MMLU_PRO", "GPQA", "CMMLU", "HalluQA-main"]
# categories = ["", "", "", "", "", "", ""]
models =  {
    "Qwen2.5-14B-instruct（阿里云）": [78.45, 82.37, 78.13, 64.02, 46.46, 80.19, 62.22],
    "DeepSeek-R1-Distill-Qwen-14B（DeepSeek）": [71.9, 67.38, 86.08, 73.23, 55.56, 81.77, 43.56],
    "Phi-4（微软）": [60.81, 63.26, 85.92, 72.9, 55.56, 65.03, 18.67],
    "InternLM2-chat-Math-20B（上海人工智能实验室）": [77.63, 80.37, 38.53, 22.11, 30.81, 61.24, 29.78],
    "Gemma3-12B（谷歌）": [81.7, 58.22, 78.54, 76.51, 36.36, 61.39, 28.67],
    "GLM-9B-chat（智谱）": [81.89, 70.67, 71.66, 46.14, 30.3, 70.39, 38.00],
    "Baichuan2-13B-Chat（百川）": [12.38, 55.43, 12.66, 16.62, 5.05, 59.93, 24.44],
    "EduChat-sft-002-13b（华东师范大学）": [35.12, 42.39, 24.61, 12.06, 12.12, 42.82, 15.56],
    "Confucius-o1（网易）": [60.26, 78.31, 74.05, 49.2, 38.38, 83.49, 54.67],
    "Spark-lite（讯飞）": [7.76, 31.83, 28.31, 17.64, 25.25, 30.16, 13.62],
    "师承万象（北京师范大学）": [75.97, 81.88, 84.66, 67.43, 45.96, 82.33, 73.67]
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
        label.set_fontsize(25)  # 增大字体大小
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
                  fontsize=20,
                  frameon=False)

# 调整布局 
plt.tight_layout()

# 保存图片
plt.savefig('../imgs/evals/radar_tongyong.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()    