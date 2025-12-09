import pyaudio
import numpy as np
import time
import sys

# --- 🎛️ 核心参数调试区 ---

# 1. 触发阈值 (ONSET): 只有能量突变量超过这个值，才判定为拨弦
#    调大 = 需要更用力拨弦才能触发
#    调小 = 更灵敏
ONSET_THRESHOLD = 2e8

# 2. 复位阈值 (RESET): 只有当前总能量低于这个值，系统才会“解锁”，准备下一次识别
#    这个值必须比拨弦时的能量低很多，但要比背景噪音高
#    如果你的弦一直在响导致无法识别下一次，请调大这个值
RESET_THRESHOLD = 8e7

# --- 其他参数 ---
CHUNK = 2048              
FORMAT = pyaudio.paInt16  
CHANNELS = 1              
RATE = 44100              

# 音名列表
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ACTION_MAP = {
    "F#3": "空格", 
    "G3":  "回车",
    "B3":  "感叹号"
}

def freq_to_note_name(freq):
    if freq == 0: return None
    try:
        n = 12 * np.log2(freq / 440.0) + 69
        n_round = int(round(n))
        note_idx = n_round % 12
        octave = (n_round // 12) - 1
        return NOTE_NAMES[note_idx] + str(octave)
    except:
        return None

def parabolic_interpolation(magnitude_spectrum, peak_idx):
    if peak_idx < 1 or peak_idx >= len(magnitude_spectrum) - 1: return peak_idx
    alpha = magnitude_spectrum[peak_idx - 1]
    beta = magnitude_spectrum[peak_idx]
    gamma = magnitude_spectrum[peak_idx + 1]
    denom = 2 * (2 * beta - gamma - alpha)
    if denom == 0: return peak_idx
    return peak_idx + 0.5 * (gamma - alpha) / denom

def detect_pitch(signal, rate):
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    hps = np.copy(spectrum)
    for h in range(2, 4):
        decimated = spectrum[::h]
        hps[:len(decimated)] *= decimated
    start_idx = int(60 * len(spectrum) / (rate / 2)) 
    peak_idx = np.argmax(hps[start_idx:]) + start_idx
    true_idx = parabolic_interpolation(hps, peak_idx)
    return true_idx * rate / CHUNK

def draw_bar(energy, is_ready):
    """在控制台绘制能量条，方便调试"""
    # 将能量对数化以便显示
    if energy < 1: energy = 1
    log_energy = np.log10(energy)
    
    # 简单的缩放映射，根据你的麦克风可能需要调整 range
    bar_len = int((log_energy - 4) * 10) 
    if bar_len < 0: bar_len = 0
    if bar_len > 50: bar_len = 50
    
    bar = "█" * bar_len + "░" * (50 - bar_len)
    
    status = "🟢 待命 (Ready)" if is_ready else "🔴 锁定 (Locked)"
    sys.stdout.write(f"\r能量: [{bar}] {int(energy)} | {status}")
    sys.stdout.flush()

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎸 严谨防抖模式启动")
    print(f"触发阈值: {ONSET_THRESHOLD} | 复位阈值: {RESET_THRESHOLD}")
    print("---------------------------------------------------")

    prev_energy = 0
    
    # 核心状态标志：是否准备好接受下一次拨弦
    # True = 安静等待中
    # False = 刚刚拨过，正在等待琴弦静止
    is_ready_to_trigger = True 

    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16)
            
            # 计算绝对总能量
            curr_energy = np.sum(data_int.astype(float)**2)
            
            # 计算能量突变量 (Flux)
            flux = curr_energy - prev_energy
            prev_energy = curr_energy 

            # --- 状态机逻辑 ---

            if is_ready_to_trigger:
                # 状态 1: 待命模式
                # 只有能量突然暴涨 (Flux > 阈值) 才会触发
                if flux > ONSET_THRESHOLD:
                    
                    # 🎯 触发识别！
                    freq = detect_pitch(data_int, RATE)
                    note = freq_to_note_name(freq)
                    
                    # 换行打印以免破坏进度条显示
                    sys.stdout.write("\n") 
                    if note:
                        action = ACTION_MAP.get(note, "")
                        if action:
                            print(f"🚀 成功触发: {note} -> {action}")
                        else:
                            print(f"   识别到: {note} (无映射)")
                    else:
                        print("   (噪音/未识别)")
                    
                    # 🔒 立即锁死系统
                    is_ready_to_trigger = False
            
            else:
                # 状态 2: 锁定模式 (Reseting)
                # 在这个模式下，无论怎么拨弦，程序都不理会
                # 只有当绝对能量 (curr_energy) 降到非常低 (RESET_THRESHOLD) 时
                # 才重新把系统设为 "Ready"
                
                if curr_energy < RESET_THRESHOLD:
                    is_ready_to_trigger = True
                    # sys.stdout.write("\n🔄 系统复位，准备下次拨弦...\n") # 调试用，嫌吵可注释

            # 绘制实时能量条 (可选，会降低一点点性能)
            draw_bar(curr_energy, is_ready_to_trigger)

    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()