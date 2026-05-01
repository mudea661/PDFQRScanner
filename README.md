# 增强版二维码识别器

> 本项目全程 AI 构建

基于 OpenCV 和 WeChatQRCode 的 PDF 二维码识别工具，支持多进程并行与实时统计。

## ✨ 功能特点

- 🌟 **双引擎识别**：集成 OpenCV QRCode 和 WeChatQRCode，提升复杂场景识别率
- ⚡ **并行处理**：支持多进程并行扫描（默认 CPU核数-1，最少1）
- 🔄 **旋转不变**：自动尝试 4 个方向识别，适应各种角度的二维码
- 📊 **精准定位**：准确识别二维码所在页面
- 🖥️ **图形界面**：简洁易用的 UI 界面
- 🧾 **双格式输出**：自动输出 JSON 和按页排序 TXT（一行一个内容）

## 🛠️ 技术栈

- **语言**: Python 3.8+
- **GUI 框架**: Tkinter
- **二维码识别**: OpenCV + WeChatQRCode
- **PDF 处理**: PyMuPDF
- **并行计算**: multiprocessing

## ⚡ 快速开始

```bash
# 克隆项目
git clone https://github.com/yourusername/PDFQRScanner.git
cd PDFQRScanner

# 安装依赖
pip install -r requirements.txt

# 下载模型文件到 opencv_3rdparty/ 目录
# 下载地址：https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode

# 运行
python enhanced_qr_ui.py
```

## 📁 文件结构

```
PDFQRScanner/
├── enhanced_qr_ui.py            # 图形界面主程序
├── opencv_3rdparty/             # WeChatQRCode 模型文件（需下载）
│   ├── detect.caffemodel
│   ├── detect.prototxt
│   ├── sr.caffemodel
│   └── sr.prototxt
├── output/                      # 识别结果输出目录
├── requirements.txt             # Python 依赖清单
└── README.md                    # 说明文档
```

## 🚀 从零到一运行步骤

### 步骤 1：安装 Python

1. 下载 Python 3.8+ 版本：https://www.python.org/downloads/
2. 安装时勾选 "Add Python to PATH"

### 步骤 2：下载模型文件（可选但推荐）

WeChatQRCode 模型文件需要单独下载，解压到 `opencv_3rdparty` 文件夹：

- **下载地址**：[WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode)
- 需要下载的文件：
  - `detect.caffemodel`（约 965KB）
  - `detect.prototxt`（约 45KB）
  - `sr.caffemodel`（约 24KB）
  - `sr.prototxt`（约 6KB）

**注意事项**：
- 只需要上述 4 个模型文件即可，不需要完整仓库历史
- 将上述 4 个文件放入 `PDFQRScanner/opencv_3rdparty/`
- 若未放置模型文件，程序会继续使用 OpenCV 原生识别

### 步骤 3：安装依赖

打开命令行，进入 PDFQRScanner 文件夹：

```bash
cd PDFQRScanner
pip install -r requirements.txt
```

### 步骤 4：运行程序

```bash
python enhanced_qr_ui.py
```

## 📖 使用说明

### 界面操作

1. **选择 PDF 文件**：点击"选择..."按钮选择要识别的 PDF 文件
2. **确认自动输出路径**：输出路径会自动生成，无需手动选择
3. **参数配置**（可选）：
   - 缩放倍数：建议 3.0（数字越大越清晰，但速度越慢）
   - 进程数：默认 CPU核数-1（最少1），建议按 CPU 核心数调整
   - 使用 WeChat 识别：启用可提高识别率
   - 尝试旋转：启用可识别旋转的二维码
4. **开始识别**：点击"开始识别"按钮
5. **查看结果**：识别完成后自动生成 JSON 与 TXT 两个文件

### 输出格式与目录规则

程序会输出到项目目录下的 `output/` 文件夹（打包运行时输出到当前运行目录下的 `output/`）。

假设输入文件为 `demo.pdf`，会生成：

- `output/demo_result.json`
- `output/demo_result_pages.txt`（按页码顺序，每行一个二维码内容）

JSON 示例：

```json
[
  {
    "page": 15,
    "content": "http://example.com/qr/abc123"
  },
  {
    "page": 42,
    "content": "WIFI:T:WPA;S:MyWiFi;P:password123;;"
  }
]
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 缩放倍数 | 图像渲染缩放倍数 | 3.0 |
| 进程数 | 并行处理进程数 | CPU核数-1（最少1） |
| 使用 WeChat 识别 | 是否启用 WeChatQRCode 引擎 | 是 |
| 尝试旋转 | 是否尝试 4 个方向识别 | 是 |

## 📊 性能指标

| 指标 | 性能 |
|------|------|
| 识别率 | 在测试样本中表现稳定（受PDF质量与二维码清晰度影响） |
| 处理速度 | 在测试样本中约 5-10 分钟/2000 页（与CPU和参数配置相关） |
| 支持格式 | PDF |

## 🛠️ 系统要求

- Windows 10/11
- Python 3.8+
- 至少 4GB 内存（推荐 8GB+）

## 📝 常见问题

**Q: 运行时提示找不到模型文件？**

A: 请确保 `opencv_3rdparty` 文件夹中包含以下 4 个文件：
   - detect.caffemodel
   - detect.prototxt
   - sr.caffemodel
   - sr.prototxt

**Q: 识别速度很慢？**

A: 可以尝试：
   - 降低缩放倍数（如从 3.0 降到 2.0）
   - 调整进程数（默认 CPU核数-1，建议按 CPU 核心数）

**Q: 识别率不高？**

A: 请确保：
   - 启用了 "使用 WeChat 识别" 选项
   - 启用了 "尝试旋转" 选项
   - 适当提高缩放倍数

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License