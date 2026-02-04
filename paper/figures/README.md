# 论文图表目录

本目录用于存放论文第三章所需的图表文件。

## 需要的图表

### 图3.1：技术路线图
- 文件名：`pipeline.pdf`
- 内容：三阶段技术路线（提示工程 → SFT → 蒸馏）
- 建议工具：draw.io, PowerPoint

### 图3.2：性能对比图
- 文件名：`comparison.pdf`
- 内容：BERT vs LLM 性能对比（F1分数）
- 生成方式：运行 `generate_paper_figures.py`

### 图3.3：学习曲线
- 文件名：`learning_curves.pdf`
- 内容：训练过程中的F1/Precision/Recall变化
- 生成方式：运行 `generate_paper_figures.py`

## 自动生成

训练完成后，运行以下命令自动生成图表：

```bash
cd paper
python generate_paper_figures.py
```

生成的PDF文件将保存在本目录中。

## 手动创建

如果需要手动创建图表：
1. 使用 `training_history.csv` 数据绘制学习曲线
2. 使用 `final_evaluation.json` 数据绘制对比图
3. 保存为PDF格式（推荐）或高分辨率PNG
