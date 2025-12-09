import pyaudio
import numpy as np
import json
import os
import time
import math
from pynput.keyboard import Controller, Key

# ==========================================
# 🔧 配置参数
# ==========================================
SAMPLE_RATE = 48000  
CHUNK_SIZE = 4096    
HOP_SIZE = 1024      

# 默认阈值 (会被自动校准覆盖)
VOLUME_THRESHOLD = 0.01 

# [冲击检测]
ATTACK_SENSITIVITY = 0.003 
DEBOUNCE_TIME = 0.12

MAPPING_FILE = "guitar_mapping.json"

class GuitarHPS:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.keyboard = Controller()
        self.mapping = {}
        self.input_device_index = None
        self.load_mapping()
        self.window = np.hanning(CHUNK_SIZE)
        self.last_spectrum = np.zeros(CHUNK_SIZE // 2 + 1)
        # 将阈值作为类属性，方便修改
        self.threshold = VOLUME_THRESHOLD 

    def select_device(self):
        print("\n=== 请选择麦克风/声卡 ===")
        cnt = self.p.get_device_count()
        valid = []
        for i in range(cnt):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"ID {i}: {info['name']}")
                valid.append(i)
        
        while True:
            try:
                sel = input(f"请输入 ID: ")
                idx = int(sel)
                if idx in valid:
                    self.input_device_index = idx; break
            except: pass

    def calibrate_noise(self):
        """★ 新功能：自动环境噪音校准"""
        stream = self.get_stream()
        print("\n=== 正在校准环境噪音 ===")
        print("🤫 请保持安静，不要弹琴，手捂住弦...")
        
        max_noise = 0
        calibration_time = 2.0 # 采样2秒
        start_time = time.time()
        
        buffer_list = []
        
        try:
            while time.time() - start_time < calibration_time:
                raw = stream.read(HOP_SIZE, exception_on_overflow=False)
                shorts = np.frombuffer(raw, dtype=np.int16)
                floats = shorts.astype(np.float32) / 32768.0 * 5.0 # 保持同样的放大倍数
                
                vol = np.sqrt(np.mean(floats**2))
                if vol > max_noise:
                    max_noise = vol
                
                # 打印进度条
                print(f"\r采样中... 当前噪音: {vol:.4f} | 峰值: {max_noise:.4f}", end='')
            
            # 设定新阈值：噪音峰值 * 1.5 (安全系数)
            # 至少保留 0.005 的底限
            new_threshold = max(max_noise * 1.5, 0.005)
            self.threshold = new_threshold
            
            print(f"\n\n✅ 校准完成！")
            print(f"检测到底噪: {max_noise:.4f}")
            print(f"已设置新门限: {self.threshold:.4f}")
            print("-" * 30)
            
        finally:
            stream.stop_stream()
            stream.close()

    def load_mapping(self):
        if os.path.exists(MAPPING_FILE):
            try:
                with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
            except: self.mapping = {}

    def save_mapping(self):
        try:
            with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=4)
        except: pass

    def get_stream(self):
        return self.p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                           input_device_index=self.input_device_index, frames_per_buffer=HOP_SIZE)

    def freq_to_note(self, freq):
        if freq < 60 or freq > 1500: return None
        name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        try:
            h = round(12 * math.log2(freq / 440.0)) + 57
            return name[h % 12] + str(h // 12)
        except: return None

    def hps_pitch_flux(self, data, is_attack_frame=False):
        windowed = data * self.window
        spec = np.abs(np.fft.rfft(windowed))
        spec[0] = 0
        target_spec = spec
        
        if is_attack_frame:
            flux = np.maximum(spec - self.last_spectrum, 0)
            if np.max(flux) > 0.1: target_spec = flux

        self.last_spectrum = spec * 0.5 + self.last_spectrum * 0.5

        hps1 = target_spec
        hps2 = target_spec[::2]
        hps3 = target_spec[::3]
        hps4 = target_spec[::4] 
        min_len = min(len(hps1), len(hps2), len(hps3), len(hps4))
        hps_spec = hps1[:min_len] * hps2[:min_len] * hps3[:min_len] * hps4[:min_len]
        
        peak_idx = np.argmax(hps_spec)
        return peak_idx * (SAMPLE_RATE / CHUNK_SIZE)

    def train_mode(self):
        stream = self.get_stream()
        print("\n=== 训练模式 ===")
        print(f"当前生效门限: {self.threshold:.4f}")
        
        buffer = np.zeros(CHUNK_SIZE) 
        self.last_spectrum = np.zeros(CHUNK_SIZE // 2 + 1)
        
        try:
            while True:
                last_n = None; stable = 0
                print("\n[等待弹奏] ... ", end='')

                while stable < 3:
                    raw = stream.read(HOP_SIZE, exception_on_overflow=False)
                    shorts = np.frombuffer(raw, dtype=np.int16)
                    floats = shorts.astype(np.float32) / 32768.0 * 5.0
                    
                    buffer = np.roll(buffer, -HOP_SIZE)
                    buffer[-HOP_SIZE:] = floats
                    vol = np.sqrt(np.mean(buffer**2))
                    
                    curr_note = None
                    # 使用校准后的 threshold
                    if vol > self.threshold:
                        freq = self.hps_pitch_flux(buffer, is_attack_frame=False)
                        curr_note = self.freq_to_note(freq)
                        d = curr_note if curr_note else "---"
                        print(f"\r[分析] Vol:{vol:.4f} | Note:{d:<4}   ", end='')
                    else:
                        # 显示当前底噪以便观察
                        print(f"\r[安静] Vol:{vol:.4f} < {self.threshold:.4f}   ", end='')
                    
                    if curr_note and curr_note == last_n: stable += 1
                    elif curr_note: last_n = curr_note; stable = 1
                
                print(f"\n✅ 锁定: 【 {last_n} 】")
                
                if last_n in self.mapping: print(f"当前绑定: {self.mapping[last_n]}")
                key = input("输入按键 (enter/space/exit): ").strip()
                
                if key == 'exit': break
                if key:
                    k = '\n' if key=='enter' else ' ' if key=='space' else key[0]
                    self.mapping[last_n] = k
                    print(f"已保存: {last_n} -> {repr(k)}")

                print("⏳ 切音...", end='')
                s_time = time.time()
                while True:
                    raw = stream.read(HOP_SIZE, exception_on_overflow=False)
                    shorts = np.frombuffer(raw, dtype=np.int16)
                    # 检查是否低于校准后的阈值
                    if np.linalg.norm(shorts) < (self.threshold * 32768 / 5.0): 
                        time.sleep(0.1); break
                    if time.time() - s_time > 3.0: break 
                print("\r🚀 就绪!    ")

        except KeyboardInterrupt: pass
        finally: self.save_mapping(); stream.stop_stream(); stream.close()

    def run_mode(self):
        if not self.mapping: print("请先训练"); return
        stream = self.get_stream()
        print("\n=== 演奏模式 ===")
        print(f"使用门限: {self.threshold:.4f}")
        
        buffer = np.zeros(CHUNK_SIZE)
        self.last_spectrum = np.zeros(CHUNK_SIZE // 2 + 1)
        
        last_vol = 0; last_time = 0
        
        try:
            while True:
                raw = stream.read(HOP_SIZE, exception_on_overflow=False)
                shorts = np.frombuffer(raw, dtype=np.int16)
                floats = shorts.astype(np.float32) / 32768.0 * 5.0 
                
                buffer = np.roll(buffer, -HOP_SIZE)
                buffer[-HOP_SIZE:] = floats
                vol = np.sqrt(np.mean(buffer**2))
                
                is_attack = (vol - last_vol) > ATTACK_SENSITIVITY
                is_cool = (time.time() - last_time) > DEBOUNCE_TIME
                
                if is_attack and is_cool:
                    freq = self.hps_pitch_flux(buffer, is_attack_frame=True)
                    curr_note = self.freq_to_note(freq)
                    
                    if curr_note:
                        if curr_note in self.mapping:
                            char = self.mapping[curr_note]
                            if char == '\n': self.keyboard.press(Key.enter); self.keyboard.release(Key.enter); d="[Enter]"
                            elif char == ' ': self.keyboard.press(Key.space); self.keyboard.release(Key.space); d="[Space]"
                            else: self.keyboard.type(char); d=char
                            print(f" >> 输入: {d} | 音: {curr_note} | 强度: {vol:.3f}")
                        else:
                            print(f" >> 未绑定: {curr_note}")
                        last_time = time.time()
                    
                elif vol > self.threshold:
                     self.hps_pitch_flux(buffer, is_attack_frame=False)

                last_vol = vol

        except KeyboardInterrupt: pass
        finally: stream.stop_stream(); stream.close()

if __name__ == "__main__":
    h = GuitarHPS()
    h.select_device()
    
    # ★ 启动流程变化
    print("\n[系统] 建议先进行噪音校准！")
    print("1. 训练模式")
    print("2. 演奏模式")
    print("3. 自动校准 (推荐)")
    
    choice = input("请选择: ")
    
    if choice == '3':
        h.calibrate_noise()
        # 校准完自动问下一步
        i = input("\n校准完毕。去哪里？(1.训练 / 2.演奏): ")
        if i == '1': h.train_mode()
        elif i == '2': h.run_mode()
    elif choice == '1':
        h.train_mode()
    elif choice == '2':
        h.run_mode()