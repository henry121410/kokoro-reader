# kokoro-reader: 选中即读 TTS 工具

`kokoro-reader` 是一个简单的桌面工具，让你可以在任何应用程序中选中文字，然后通过按下一个全局快捷键，就能听到用 `kokoro` TTS 引擎朗读出来的声音。

## 主要功能

*   **选中即读**: 在浏览器、PDF、编辑器等任何地方选中文字即可朗读。
*   **全局快捷键**: 通过自定义的快捷键（默认是 `Ctrl+~`，）触发朗读，无需切换窗口。
*   **高质量语音**: 使用 [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS 引擎生成自然流畅的语音。
*   **后台运行**: 程序在后台静默运行，不干扰你的工作。

## 快速开始 (开发中)

**(注意：以下步骤是基于项目规划，具体实现可能需要调整)**

1.  **安装依赖**: 
    确保你安装了 Python 3.9+ 和 Git。
    ```bash
    # 克隆项目 (如果你还没有)
    git clone https://github.com/thewh1teagle/kokoro-reader.git
    cd kokoro-reader
    
    # 安装 Python 依赖
    pip install -r requirements.txt 
    
    # 安装 espeak-ng (kokoro 可能需要)
    # Windows 用户请参照下面的 "Windows 安装 espeak-ng" 指南
    # Linux (Debian/Ubuntu): 
    # sudo apt-get update && sudo apt-get install espeak-ng
    # macOS:
    # brew install espeak-ng 
    ```
    *(我们稍后会创建 `requirements.txt` 文件)*

2.  **运行程序**:
    ```bash
    python src/main.py  # 或者 python main.py 如果主脚本在根目录
    ```
    *(我们稍后会创建 `main.py` 文件)*

3.  **使用**: 
    *   程序启动后会在后台运行。
    *   在任何地方选中你想要朗读的文本。
    *   按下设定的全局快捷键 (例如 `Ctrl+~`)。
    *   稍等片刻，你应该就能听到朗读的声音。

## Windows 安装 espeak-ng

`kokoro` 引擎可能依赖 `espeak-ng` 来处理某些语言或作为备选方案。在 Windows 上安装它：

1.  访问 [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)。
2.  找到最新的稳定版本 (Latest release)。
3.  下载适合你系统的 `.msi` 安装包 (例如 `espeak-ng-xxxxxxxx-x64.msi` 代表 64 位系统)。
4.  运行下载的 `.msi` 文件进行安装。

## 项目文档

更详细的项目规划、需求、技术细节等，请参考 `project_docs/` 目录下的文档。

## 贡献 (开发中)

欢迎提出建议和贡献代码！(详细贡献指南待补充)

## 致谢

*   本项目使用了 [kokoro](https://github.com/thewh1teagle/kokoro) TTS 库。
*   感谢 `note.md` 中提供的初始想法和技术方案。
