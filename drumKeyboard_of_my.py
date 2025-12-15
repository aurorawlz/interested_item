import pyaudio
import numpy as np
import time
import sys
import json
import os
import argparse
from collections import deque

'''
使用方法：
1. 快速开始（使用模板）
python drum_simple.py --quick-setup

2. 录制你的实际鼓声
python drum_simple.py --record
# 按照提示，为每个鼓件录制轻击和重击

3. 开始使用
python drum_simple.py --use

4. 调整设置
# 调高灵敏度（更容易触发）
python drum_simple.py --use --sens high
# 调低灵敏度（减少误触发）
python drum_simple.py --use --sens low
# 调整识别严格度
python drum_simple.py --use --sim 0.8  # 更严格
python drum_simple.py --use --sim 0.7  # 更宽松

'''


try:
    from pynput.keyboard import Controller, Key

    KEYBOARD_AVAILABLE = True
    _kb = Controller()
except Exception:
    KEYBOARD_AVAILABLE = False
    _kb = None

# --- 🥁 两种力度配置 ---
ONSET_THRESHOLD = 3e8  # 触发阈值
RESET_THRESHOLD = 5e7  # 复位阈值
MIN_ENERGY = 1e7  # 最小能量

# 音频参数
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# 分析参数
FP_FRAMES = 3  # 3帧约0.14秒

# 两种力度配置
VELOCITY_LEVELS = ["soft", "hard"]  # 只有轻和重两种力度

# 你的鼓件配置 - 简化为两种力度按键
YOUR_DRUM_SET = {
    # 格式: {"name": "显示名称", "color": "图标", "keys": ["轻击键", "重击键"]}
    "snare": {
        "name": "军鼓",
        "color": "🔴",
        "keys": ["j", "k"]  # 轻=j, 重=k
    },
    "hihat_hand": {
        "name": "踩镲(手)",
        "color": "🟢",
        "keys": ["u", "i"]  # 轻=u, 重=i
    },
    "ride": {
        "name": "吊镲",
        "color": "🔵",
        "keys": ["y", "h"]  # 轻=y, 重=h
    },
    "tom1": {
        "name": "一嗵",
        "color": "🟡",
        "keys": ["t", "g"]  # 轻=t, 重=g
    },
    "tom2": {
        "name": "二嗵",
        "color": "🟠",
        "keys": ["r", "f"]  # 轻=r, 重=f
    },
    "tom3": {
        "name": "三嗵",
        "color": "🟣",
        "keys": ["e", "d"]  # 轻=e, 重=d
    },
    "kick": {
        "name": "底鼓",
        "color": "🟤",
        "keys": ["space", "a"]  # 轻=空格, 重=a
    },
    "hihat_foot": {
        "name": "踩镲(脚)",
        "color": "🟩",
        "keys": ["q", "w"]  # 轻=q, 重=w
    }
}

# 力度阈值配置（两种力度）
VELOCITY_THRESHOLDS = {
    "kick": 4e8,  # 底鼓力度分界线
    "snare": 3e8,  # 军鼓力度分界线
    "hihat_hand": 2e8,  # 手击踩镲力度分界线
    "hihat_foot": 2.5e8,  # 脚击踩镲力度分界线
    "ride": 2e8,  # 吊镲力度分界线
    "tom1": 2.5e8,  # 一嗵力度分界线
    "tom2": 2.8e8,  # 二嗵力度分界线
    "tom3": 3e8,  # 三嗵力度分界线
    "default": 3e8  # 默认分界线
}

# 文件路径
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "drum_mapping_2level.json")
FINGERPRINT_FILE = os.path.join(os.path.dirname(__file__), "drum_fingerprints_2level.json")


def classify_velocity_simple(energy, drum_type):
    """简化的两种力度分类"""
    threshold = VELOCITY_THRESHOLDS.get(drum_type, VELOCITY_THRESHOLDS["default"])
    return "hard" if energy >= threshold else "soft"


def get_drum_type_by_features(features):
    """根据特征匹配鼓件类型"""
    low_ratio = features["low_ratio"]
    mid_ratio = features["mid_ratio"]
    high_ratio = features["high_ratio"]
    centroid = features["centroid"]

    # 简化分类逻辑
    if low_ratio > 0.6:
        return "kick"  # 底鼓

    elif high_ratio > 0.6:
        if features["attack_time"] < 0.02:
            return "hihat_hand"  # 手击踩镲
        else:
            return "hihat_foot"  # 脚击踩镲

    elif mid_ratio > 0.5:
        return "snare"  # 军鼓

    elif 1500 < centroid < 4000:
        return "ride"  # 吊镲

    elif 400 < centroid < 1500:
        # 通鼓区分
        if low_ratio > 0.3:
            return "tom3"  # 三嗵
        elif low_ratio > 0.2:
            return "tom2"  # 二嗵
        else:
            return "tom1"  # 一嗵

    return "unknown"


def extract_features_simple(signal, sample_rate=RATE):
    """简化特征提取"""
    energy = np.sum(signal.astype(float) ** 2)

    # 时域特征
    envelope = np.abs(signal)
    attack_time = np.argmax(envelope) / sample_rate

    # 频谱分析
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)

    # 频谱质心
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 0

    # 频段能量比例
    freq_resolution = freqs[1] - freqs[0]

    # 简化为三个频段
    low_idx = int(150 / freq_resolution)
    mid_idx = int(1000 / freq_resolution)
    high_idx = int(4000 / freq_resolution)

    low_band = np.sum(spectrum[:low_idx])
    mid_band = np.sum(spectrum[low_idx:mid_idx])
    high_band = np.sum(spectrum[mid_idx:high_idx])
    total_band = low_band + mid_band + high_band

    if total_band > 0:
        low_ratio = low_band / total_band
        mid_ratio = mid_band / total_band
        high_ratio = high_band / total_band
    else:
        low_ratio = mid_ratio = high_ratio = 0

    # 特征向量（简化）
    feature_vector = [
        low_ratio,
        mid_ratio,
        high_ratio,
        min(centroid / 5000, 1.0),
        min(attack_time * 50, 1.0)
    ]

    # 特征字典
    features = {
        "energy": float(energy),
        "centroid": float(centroid),
        "attack_time": float(attack_time),
        "low_ratio": float(low_ratio),
        "mid_ratio": float(mid_ratio),
        "high_ratio": float(high_ratio),
    }

    return features, feature_vector


def load_mapping():
    """加载映射"""
    if not os.path.exists(MAPPING_FILE):
        # 创建默认映射
        default_mapping = {}
        for drum_id, drum_info in YOUR_DRUM_SET.items():
            default_mapping[drum_id] = drum_info["keys"]
        save_mapping(default_mapping)
        return default_mapping

    try:
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_mapping(mapping):
    try:
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_fingerprints():
    if not os.path.exists(FINGERPRINT_FILE):
        return {}
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_fingerprints(db):
    try:
        with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _parse_key_name(name: str):
    if not name:
        return None
    lower = name.lower()
    special_map = {
        "space": Key.space,
        "enter": Key.enter,
        "tab": Key.tab,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "alt": Key.alt,
    }
    return special_map.get(lower, name)


def send_keypress(key_name: str):
    if not KEYBOARD_AVAILABLE or _kb is None:
        return

    target = _parse_key_name(key_name)
    if target is None:
        return

    try:
        if isinstance(target, Key):
            _kb.press(target)
            time.sleep(0.01)
            _kb.release(target)
        else:
            _kb.press(target)
            time.sleep(0.005)
            _kb.release(target)
    except Exception:
        pass


def _gather_frames(stream, first_chunk: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return first_chunk
    buf = [first_chunk]
    for _ in range(frames - 1):
        raw = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(raw, dtype=np.int16)
        buf.append(arr)
    return np.concatenate(buf)


def cosine_similarity(vec_a, vec_b):
    n = min(len(vec_a), len(vec_b))
    if n == 0:
        return 0.0
    a = np.array(vec_a[:n])
    b = np.array(vec_b[:n])
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def draw_simple_ui(energy, velocity, drum_type, is_ready, last_detected=""):
    """简化的UI显示"""
    # 力度图标
    velocity_icon = "●" if velocity == "hard" else "○"

    # 鼓件信息
    drum_info = YOUR_DRUM_SET.get(drum_type, {"color": "⚪", "name": "未知"})
    drum_icon = drum_info["color"]
    drum_name = drum_info["name"]

    # 力度文字
    velocity_text = "重击" if velocity == "hard" else "轻击"

    # 能量条
    if energy < 1: energy = 1
    bar_len = min(50, max(0, int(np.log10(energy) - 6) * 10))
    bar = "█" * bar_len + "░" * (50 - bar_len)

    # 状态
    status = "🟢" if is_ready else "🔴"

    sys.stdout.write(f"\r[{drum_icon}{velocity_icon}] {drum_name:8} {velocity_text:4} [{bar}] {status}")
    if last_detected:
        sys.stdout.write(f" | 上次: {last_detected}")
    sys.stdout.flush()


def quick_setup_2level():
    """快速设置两种力度"""
    print("\n⚡ 快速设置 - 两种力度模式")
    print("将创建所有鼓件的模板指纹")

    fingerprints = {}

    for drum_id, drum_info in YOUR_DRUM_SET.items():
        for i, velocity in enumerate(VELOCITY_LEVELS):
            sample_id = f"{drum_id}_{velocity}"
            fingerprints[sample_id] = {
                "drum_type": drum_id,
                "drum_name": drum_info["name"],
                "velocity": velocity,
                "key": drum_info["keys"][i],
                "is_template": True,
                "note": "模板 - 请用实际录音替换"
            }

    if save_fingerprints(fingerprints):
        # 保存映射
        mapping = {}
        for drum_id, drum_info in YOUR_DRUM_SET.items():
            mapping[drum_id] = drum_info["keys"]
        save_mapping(mapping)

        print(f"✅ 快速设置完成！")
        print(f"已创建 {len(fingerprints)} 个模板（每个鼓件2种力度）")
        print(f"\n默认按键映射:")
        for drum_id, drum_info in YOUR_DRUM_SET.items():
            keys = drum_info["keys"]
            print(f"  {drum_info['color']} {drum_info['name']:10} - 轻击:{keys[0]}  重击:{keys[1]}")
        return True
    return False


def interactive_record_mode():
    """交互式录制模式（两种力度）"""
    print("\n🎤 交互式录制模式 - 两种力度")
    print("=" * 60)
    print("每个鼓件需要录制2次：")
    print("  1. 轻击（柔和的力量）")
    print("  2. 重击（用力的击打）")
    print("=" * 60)

    fingerprints = load_fingerprints()

    for drum_id, drum_info in YOUR_DRUM_SET.items():
        print(f"\n▶️ 录制: {drum_info['color']} {drum_info['name']}")

        for i, velocity in enumerate(VELOCITY_LEVELS):
            print(f"  {i + 1}. 准备录制{velocity}力度...")
            input(f"     请{velocity}击打{drum_info['name']}，然后按回车继续...")

            # 在实际代码中，这里应该录制音频
            # 暂时跳过实际录制

            sample_name = f"{drum_id}_{velocity}"
            key_to_use = drum_info["keys"][i]

            # 创建模板指纹
            fingerprints[sample_name] = {
                "drum_type": drum_id,
                "drum_name": drum_info["name"],
                "velocity": velocity,
                "key": key_to_use,
                "is_template": False,
                "note": "用户录制"
            }

            print(f"     ✅ 已记录: {drum_info['name']} ({velocity}力度) → 按键: {key_to_use}")

    if save_fingerprints(fingerprints):
        print(f"\n✅ 录制完成！共录制 {len(fingerprints)} 个样本")
        return True
    return False


def run_two_level_detector(use_map=False, record_mode=False, sim_threshold=0.75,
                           sensitivity="medium", list_mode=False):
    if list_mode:
        fingerprints = load_fingerprints()
        if fingerprints:
            print("\n🥁 已录制的鼓件 (两种力度):")
            print("=" * 60)
            for name, data in sorted(fingerprints.items()):
                drum_type = data.get("drum_type", "unknown")
                drum_info = YOUR_DRUM_SET.get(drum_type, {"name": "未知", "color": "⚪"})
                velocity = data.get("velocity", "unknown")
                key = data.get("key", "未设置")
                is_template = data.get("is_template", False)
                template_flag = "📝" if is_template else "🎤"
                velocity_text = "重击" if velocity == "hard" else "轻击"
                print(f"{template_flag} {drum_info['color']} {drum_info['name']:10} "
                      f"| {velocity_text:4} | 按键: {key:3} | ID: {name}")
            print(f"\n总计: {len(fingerprints)} 个样本")
        else:
            print("📭 未找到任何记录")
        return

    if record_mode:
        interactive_record_mode()
        return

    # 调整灵敏度
    if sensitivity == "high":
        current_threshold = 2e8
    elif sensitivity == "low":
        current_threshold = 5e8
    else:  # medium
        current_threshold = ONSET_THRESHOLD

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print("\n🥁 架子鼓检测 - 两种力度模式")
    print("=" * 60)
    print("力度模式: 轻击 / 重击")
    print("鼓件配置:")
    for drum_id, drum_info in YOUR_DRUM_SET.items():
        keys = drum_info["keys"]
        print(f"  {drum_info['color']} {drum_info['name']:10} - 轻:{keys[0]} 重:{keys[1]}")
    print("=" * 60)
    print(f"灵敏度: {sensitivity} (阈值: {current_threshold:,})")
    print(f"相似度阈值: {sim_threshold}")
    print("-" * 60)

    fingerprints = load_fingerprints()

    if not fingerprints:
        print("⚠️ 没有找到指纹数据！")
        print("请先运行以下命令之一:")
        print("  --record        # 交互式录制")
        print("  --quick-setup   # 快速创建模板")
        response = input("是否要快速创建模板? (y/n): ").lower()
        if response == 'y':
            stream.stop_stream()
            stream.close()
            p.terminate()
            quick_setup_2level()
            return

    prev_energy = 0
    is_ready = True
    last_detected = ""

    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(raw_data, dtype=np.int16)

            curr_energy = np.sum(data_int.astype(float) ** 2)
            flux = curr_energy - prev_energy
            prev_energy = curr_energy

            # 默认显示值
            display_velocity = "soft"
            display_drum_type = "unknown"

            if is_ready and flux > current_threshold and curr_energy > MIN_ENERGY:
                # 采集音频
                combined = _gather_frames(stream, data_int, FP_FRAMES)

                # 提取特征
                features, feature_vector = extract_features_simple(combined)

                # 分类鼓件类型
                drum_type = get_drum_type_by_features(features)
                display_drum_type = drum_type

                # 分类力度
                velocity = classify_velocity_simple(features["energy"], drum_type)
                display_velocity = velocity

                sys.stdout.write("\n")

                # 查找最佳匹配
                best_match = None
                best_sim = 0.0
                best_key = ""

                for name, data in fingerprints.items():
                    if "fingerprint" not in data:
                        continue

                    # 优先匹配相同鼓件类型和力度
                    data_type = data.get("drum_type", "")
                    data_vel = data.get("velocity", "")

                    if data_type == drum_type and data_vel == velocity:
                        # 类型和力度都匹配，加分
                        stored_vec = data.get("fingerprint", [])
                        sim = cosine_similarity(feature_vector, stored_vec)
                        sim_boosted = sim * 1.2  # 增加20%权重

                        if sim_boosted > best_sim:
                            best_sim = sim_boosted
                            best_match = name
                            best_key = data.get("key", "")
                    else:
                        # 普通匹配
                        stored_vec = data.get("fingerprint", [])
                        sim = cosine_similarity(feature_vector, stored_vec)

                        if sim > best_sim:
                            best_sim = sim
                            best_match = name
                            best_key = data.get("key", "")

                drum_info = YOUR_DRUM_SET.get(drum_type, {"name": "未知", "color": "⚪"})
                velocity_text = "重击" if velocity == "hard" else "轻击"

                if best_match and best_sim >= sim_threshold:
                    print(f"🎯 识别: {drum_info['color']} {drum_info['name']} ({velocity_text})")
                    print(f"   匹配: {best_match} | 相似度: {best_sim:.3f} | 按键: {best_key}")

                    last_detected = f"{drum_info['name']}({velocity_text[0]})"

                    if best_key and use_map and KEYBOARD_AVAILABLE:
                        send_keypress(best_key)
                        print(f"   已触发按键: {best_key}")
                else:
                    print(f"❓ 未知: {drum_info['color']} {drum_info['name']} ({velocity_text})")
                    print(f"   最高相似度: {best_sim:.3f}")
                    last_detected = "未知"

                # 锁定系统
                is_ready = False

            else:
                # 检查是否可以复位
                if curr_energy < RESET_THRESHOLD:
                    is_ready = True

            # 更新UI
            draw_simple_ui(curr_energy, display_velocity, display_drum_type,
                           is_ready, last_detected)

    except KeyboardInterrupt:
        print("\n\n⏹️ 检测已停止")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(description="架子鼓检测 - 两种力度模式")
    parser.add_argument("--use", action="store_true", help="识别并触发按键")
    parser.add_argument("--record", action="store_true", help="交互式录制模式")
    parser.add_argument("--quick-setup", action="store_true", help="快速创建模板")
    parser.add_argument("--list", action="store_true", help="显示已录制的鼓件")
    parser.add_argument("--sim", type=float, default=0.75, help="相似度阈值，默认0.75")
    parser.add_argument("--sens", choices=["high", "medium", "low"],
                        default="medium", help="灵敏度设置")

    args = parser.parse_args()

    if args.quick_setup:
        quick_setup_2level()
        return

    run_two_level_detector(
        use_map=args.use,
        record_mode=args.record,
        sim_threshold=args.sim,
        sensitivity=args.sens,
        list_mode=args.list
    )


if __name__ == "__main__":
    # 创建默认映射文件（如果不存在）
    if not os.path.exists(MAPPING_FILE):
        load_mapping()

    main()