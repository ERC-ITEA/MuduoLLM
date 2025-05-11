import matplotlib.pyplot as plt
import numpy as np

# 定义维度和模型数据
categories = [
    '图形与几何（客观题）', '数与代数（客观题）', '统计与概率（客观题）', '综合与实践（客观题）',
    '图形与几何（主观题）', '数与代数（主观题）', '统计与概率（主观题）', '综合与实践（主观题）'
]
models = {
    'DeepSeek-R1': [0.895, 0.955, 0.915, 0.93, 0.95, 0.96, 0.94, 0.89],
    'o3-mini': [0.895, 0.92, 0.88, 0.905, 0.95, 0.96, 0.93, 0.85],
    'DeepSeek-V3': [0.89, 0.93, 0.905, 0.915, 0.95, 0.95, 0.93, 0.88],
    'GPT-4.1': [0.825, 0.81, 0.85, 0.84, 0.9, 0.94, 0.94, 0.85],
    'DeepSeek-Qwen': [0.785, 0.84, 0.755, 0.75, 0.82, 0.86, 0.82, 0.73],
    'Qwen-14B': [0.775, 0.82, 0.78, 0.71, 0.81, 0.77, 0.8, 0.72],
    'GLM-4-9B': [0.4, 0.475, 0.405, 0.37, 0.74, 0.76, 0.76, 0.6],
    'phi-4': [0.575, 0.68, 0.655, 0.455, 0.81, 0.8, 0.85, 0.73],
    'Baichuan2-13B': [0.365, 0.395, 0.375, 0.29, 0.31, 0.44, 0.33, 0.33],
    'InternLM2-20B': [0.39, 0.47, 0.43, 0.37, 0.49, 0.42, 0.43, 0.28],
    'Gemma-3-12B': [0.625, 0.67, 0.655, 0.545, 0.84, 0.83, 0.83, 0.69],
    'Confucius': [0.815, 0.825, 0.79, 0.695, 0.75, 0.82, 0.71, 0.68],
    'EduChat-Base': [0.125, 0.155, 0.155, 0.14, 0.14, 0.09, 0.3, 0.11],
    'EduChat-SFT': [0.165, 0.22, 0.12, 0.17, 0.15, 0.14, 0.27, 0.16],
    'pt_0507': [0.8, 0.815, 0.775, 0.75, 0.85, 0.88, 0.87, 0.77]
}
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] 
plt.rcParams['axes.unicode_minus'] = False  

# 设置雷达图角度
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 绘制雷达图
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'polar': True})
plt.xticks(angles[:-1], categories, fontsize=10, color='grey')


for idx, (model, scores) in enumerate(models.items()):
    scores += scores[:1] 
    ax.plot(angles, scores, linewidth=2, label=model)
    ax.fill(angles, scores, alpha=0.1)

# 设置图形标题和图例
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()
plt.savefig('/mnt/pfs_l2/jieti_team/MMGroup/lzc/SCWX_LM/imgs/evals/数学基础能力.png', 
                dpi=300, bbox_inches='tight')