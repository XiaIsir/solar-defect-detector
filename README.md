# Solar Defect Detector — 太阳能电池板缺陷检测系统

基于改进 YOLOv11n (FAF / L-FAF) 算法的太阳能电池板表面缺陷检测软件，配备 PyQt5 图形界面，支持图像、视频、摄像头实时检测。

---

## 缺陷类型

| 类别 | 英文名称 | 说明 |
|------|----------|------|
| 黑边 | Black Border | 光伏板黑色/灰色边框区域异常 |
| 隐裂 | Broken | 光伏板破损裂纹 |
| 热点 | Hot Spot | 光伏板明显亮斑区域 |
| 无电 | No Electricity | 光伏板断格/无电发黑 |
| 划痕 | Scratch | 光伏板表面划伤 |

## 快速启动

### 方式一：运行已构建的 EXE（推荐）

```
dist/solar-defect-detector/
├── solar-defect-detector.exe  主程序
├── 启动.bat                   快捷启动脚本
├── config/
│   └── users.json             用户数据（自动生成）
└── _internal/                 运行时依赖
    └── weights/
        └── best.pt            预训练模型权重
```

双击 `启动.bat` 或 `solar-defect-detector.exe` 即可运行。

首次使用需注册账号（密码至少 3 位），登录后进入主界面。

### 方式二：源码运行

```bash
conda activate fasternet
pip install -r requirements.txt
python gui_app.py
```

## 操作指南

### 1. 登录 / 注册
- 启动后弹出登录对话框
- 新用户点击「注册 | Register」创建账号
- 默认管理员账号：`admin`（密码 `123456`，SHA256 哈希存储）

### 2. 加载模型
- 程序启动时自动加载 `weights/best.pt`
- 如需切换模型，点击左侧「加载模型」按钮选择 `.pt` 文件

### 3. 选择检测模式

| 模式 | 说明 |
|------|------|
| **图片检测** | 选择单张图片或批量选择文件夹，支持 jpg/png/bmp/tif |
| **视频检测** | 选择 mp4/avi 等视频文件进行逐帧检测 |
| **摄像头检测** | 连接 USB 摄像头或 RTSP 工业相机实时检测 |

### 4. 进行检测
- **图片模式**：点击「选择文件」选取图片，点击「开始检测」
- **视频/摄像头模式**：点击「开始检测」进入连续检测，可随时点击「停止检测」

### 5. 查看结果
- 左侧显示原始图像，右侧显示带标注框的检测结果
- 下方结果表列出每个检测框的类别、置信度和坐标
- 支持缩放/平移查看细节
- 置信度滑块可动态过滤低置信度检测结果

### 6. 导出结果
- 点击「导出结果」将当前检测结果保存为带标注的图像
- 支持 CSV 格式导出检测数据

## 数据集

数据集位于 `datasets/solar_dataset/`：

```
datasets/solar_dataset/
├── images/
│   ├── train/    884 张训练图片
│   └── val/      222 张验证图片
└── labels/
    ├── train/
    └── val/
```

来源：PV-Multi-Defect 数据集（CC BY 3.0），论文 GBH-YOLOv5 (Electronics, 2023)。

训练配置：`data.yaml`

## 模型架构

提供两种改进算法：

### FAF-YOLOv11n（精度优先）
- 引入 FEM 自适应多尺度特征处理卷积模块
- C3k2_AGA 代理注意力机制，线性复杂度全局建模
- 参数量 3.88M，适合对精度要求高的检测场景

### L-FAF-YOLOv11n（轻量优先）
- EFBlock 轻量化模块 + PConv 部分卷积
- BiFPN 双向特征金字塔自适应加权融合
- 参数量 2.89M，适合边缘端/实时部署

模型配置：`ultralytics/cfg/models/11/l_faf_yolov11n.yaml`

## 项目结构

```
├── gui_app.py                         PyQt5 图形界面（登录、检测、结果展示）
├── data.yaml                          数据集配置
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── block.py               自定义算子：ELA、PConv、C3k2_AGA、FEM 等
│   │   │   └── __init__.py            模块导出
│   │   └── tasks.py                   模型解析器 parse_model（自定义模块路由）
│   └── cfg/models/11/
│       └── l_faf_yolov11n.yaml        L-FAF 网络拓扑配置
├── datasets/solar_dataset/            太阳能板缺陷数据集
├── dist/solar-defect-detector/        部署构建目录
└── scripts/
    └── train_l_faf_intel.py           训练脚本
```

## 开发指南

### 环境要求
- Python 3.10+
- PyTorch (>=2.0)
- Ultralytics YOLO
- PyQt5

### 重新打包 EXE

```bash
conda activate fasternet
pip install pyinstaller
pyinstaller solar-defect-detector.spec
```

### 自定义模块开发

所有自定义 PyTorch 模块实现在 `ultralytics/nn/modules/block.py`，需在 `__init__.py` 的 `__all__` 中暴露，并在 `tasks.py` 的 `parse_model` 中添加路由逻辑。

详细规约见 `CLAUDE.md`。

## 技术栈

- **深度学习框架**: PyTorch + Ultralytics YOLO
- **图像处理**: OpenCV
- **GUI 框架**: PyQt5
- **打包部署**: PyInstaller
