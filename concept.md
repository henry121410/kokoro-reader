import time
import threading
import pyperclip
from pynput import keyboard, mouse # 需要 pynput 来模拟按键
# 假设你已经设置好了 kokoro_onnx_tts 函数和 play_audio 函数
# from your_tts_module import kokoro_onnx_tts, play_audio

# --- 配置 ---
HOTKEY = {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char('s')} # Ctrl+Shift+S
current_keys = set()
processing_lock = threading.Lock() # 防止并发处理

# --- 模拟按键控制器 ---
key_controller = keyboard.Controller()

# --- TTS 和播放函数 (需要你自己实现) ---
def kokoro_onnx_tts(text):
    """
    调用 kokoro-onnx 生成音频数据 (例如 NumPy 数组)
    """
    print(f"正在合成文本: {text[:30]}...")
    # 这里替换为实际的 kokoro-onnx 调用
    # 示例：返回一个假的音频数据
    import numpy as np
    sample_rate = 22050 # 假设采样率
    duration = max(1, len(text) // 10) # 估算时长
    dummy_audio = np.random.uniform(-0.5, 0.5, size=(duration * sample_rate,)).astype(np.float32)
    print("合成完成")
    return dummy_audio, sample_rate

def play_audio(audio_data, sample_rate):
    """
    播放音频数据
    """
    print("正在播放音频...")
    try:
        import sounddevice as sd
        sd.play(audio_data, sample_rate)
        sd.wait() # 等待播放完成
        print("播放结束")
    except Exception as e:
        print(f"播放音频时出错: {e}")
        print("请确保安装了 sounddevice 并且音频设备可用。")

# --- 核心处理函数 ---
def process_selected_text():
    """
    模拟复制、获取文本、进行 TTS 和播放
    """
    if not processing_lock.acquire(blocking=False):
        print("正在处理上一个请求，请稍候...")
        return

    print("触发热键，尝试处理...")
    original_clipboard_content = pyperclip.paste() # 保存原始剪贴板内容

    try:
        # 模拟按下 Ctrl+C
        print("模拟按下 Ctrl+C")
        with key_controller.pressed(keyboard.Key.ctrl):
             key_controller.press('c')
             key_controller.release('c')

        # 等待剪贴板更新 (这个时间可能需要调整)
        time.sleep(0.1)

        selected_text = pyperclip.paste()
        print(f"获取到剪贴板内容: {selected_text[:50]}...") # 打印前50个字符

        # 恢复原始剪贴板内容 (可选，但建议)
        # pyperclip.copy(original_clipboard_content)
        # 注意：如果TTS处理时间很长，立即恢复可能会导致用户在TTS期间无法使用自己复制的内容。
        # 可以在播放结束后再恢复，或者提供选项。
        # 为了简单起见，暂时注释掉恢复步骤。

        if selected_text and selected_text != original_clipboard_content:
            print("内容有效，开始 TTS...")
            # 在新线程中执行 TTS 和播放，避免阻塞监听器
            audio_data, sr = kokoro_onnx_tts(selected_text)
            if audio_data is not None:
                 # 使用线程播放，防止阻塞热键监听
                 playback_thread = threading.Thread(target=play_audio, args=(audio_data, sr))
                 playback_thread.start()
            else:
                print("TTS 未返回有效的音频数据。")
        else:
            print("剪贴板内容未变化或为空。")

    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        processing_lock.release()
        print("处理函数执行完毕")


# --- 热键监听回调 ---
def on_press(key):
    # 将按下的键添加到集合中
    if key in HOTKEY:
        current_keys.add(key)
        # 检查是否所有热键都被按下
        if all(k in current_keys for k in HOTKEY):
            # 启动处理函数
            process_selected_text()

def on_release(key):
    # 从集合中移除释放的键
    try:
        current_keys.remove(key)
    except KeyError:
        pass # 如果键不在集合中，忽略

# --- 启动监听 ---
def start_listener():
    print(f"开始监听热键: {HOTKEY}")
    # 使用 non-blocking 监听器
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join() # 保持主线程运行，等待监听器结束

if __name__ == "__main__":
    # 确保 TTS 模型等已加载 (如果需要预加载)
    print("程序启动，准备监听热键...")
    # 在单独的线程中运行监听器，这样主线程可以做其他事情（如图形界面）
    # listener_thread = threading.Thread(target=start_listener, daemon=True)
    # listener_thread.start()
    # # 让主线程保持运行，例如可以通过 input() 或 GUI 事件循环
    # input("按 Enter 键退出...\n")

    # 或者直接在主线程运行监听器（对于简单后台应用）
    start_listener()
