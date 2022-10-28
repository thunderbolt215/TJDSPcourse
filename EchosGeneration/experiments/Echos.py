import pyaudio
import wave
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile


def save_wave_file(filename, data):
    with wave.open(filename, 'wb') as wf:  # 打开文件
        wf.setnchannels(channels)  # 设置声道数量
        wf.setsampwidth(sampwidth)  # 设置样本宽度
        wf.setframerate(framerate)  # 设置帧速率
        wf.writeframes(b"".join(data))


def my_record(filename):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1,  # 创建
                     rate=framerate, input=True,
                     frames_per_buffer=num_samples)
    my_buf = []
    print("* recording starts")
    for count in tqdm(range(0, time * 8)):  # 控制录音时间
        string_audio_data = stream.read(num_samples)
        my_buf.append(string_audio_data)
    print("* done recording")

    save_wave_file(filename + ".wav", my_buf)
    stream.close()


def plot_wave(wavfilepath, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
    # 打开WAV文档
    f = wave.open(wavfilepath, "rb")
    # 读取格式信息
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # nchannels通道数
    # sampwidth量化位数
    # framerate采样频率
    # nframes采样点数

    # 读取读取nframes个数据，返回字符串格式
    str_data = f.readframes(nframes)
    f.close()

    # 将波形数据转换为数组
    wave_data = np.frombuffer(str_data, dtype=np.short)
    # 赋值的归一化
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    # 整合左声道和右声道的数据
    wave_data = np.reshape(wave_data, [nframes, nchannels])
    # 最后通过采样点数和取样频率计算出每个取样的时间
    time = np.arange(0, nframes) * (1.0 / framerate)

    # 录制的是单声道文件
    plt.figure()
    plt.plot(time, wave_data[:, 0])
    plt.xlabel("时间t(s)", fontsize=14)
    plt.ylabel("幅度", fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid()  # 标尺

    plt.tight_layout()  # 紧密布局
    plt.show()


def playwav(wavfilepath):
    f = wave.open(wavfilepath, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()), channels=f.getnchannels(),
                    rate=f.getframerate(), output=True)
    data = f.readframes(nframes)  # 读取数据

    print("* playing starts")
    for count in tqdm(range(0, time * 8)):
        stream.write(data)
        data = f.readframes(nframes)
    print("* done playing")
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio


if __name__ == '__main__':
    # 参数设置
    framerate = 16000  # 帧速率
    num_samples = 2000  # 每个缓冲区的帧数
    channels = 1  # 声道
    sampwidth = 2  # 设置样本宽度
    time = 5  # 录制时长

    # 录制
    filepath = "origin.wav"
    my_record("origin")

    # 采样量化,输出波形
    origin_sr = librosa.get_samplerate(filepath)  # 原始采样率
    data_origin, _ = librosa.load(filepath, sr=origin_sr)  # 原始音频时间序列
    plot_wave(filepath, "原始音频")
    playwav(filepath)
    plt.show()

    # 降低一倍采样率
    data_downsample, _ = librosa.load(filepath, sr=origin_sr // 2)
    soundfile.write("downsample.wav", data_downsample, origin_sr // 2)
    plot_wave("downsample.wav", "降采样音频")
    playwav("downsample.wav")
    plt.show()

    # 回声计算
    length = len(data_origin)
    delay_time = 0.4  # 延迟时间
    decay = 0.4  # 回声衰减
    delay = int(delay_time / time * length)  # 延迟的序列单元数
    tmp1 = [data_origin[i] if i < length else 0 for i in range(0, delay + length)]
    tmp2 = np.roll(tmp1, delay)
    data_echos = list(np.array(tmp1) + np.array(tmp2))  # 计算回声音频序列
    soundfile.write("echos.wav", data_echos, origin_sr)
    plot_wave("echos.wav", "回声音频")
    playwav("echos.wav")
    plt.show()
