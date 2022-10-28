import pyaudio
import wave
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
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
    # 打开WAV文档
    f = wave.open(wavfilepath, "rb")
    # 读取格式信息
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

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

    plt.plot(time, wave_data[:, 0])
    plt.xlabel("time(s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid()


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


def plot_freq(data, title):
    D = np.abs(librosa.stft(data))
    Freq = np.arange(0, len(D))
    plt.plot(Freq, D, color='blue')
    plt.xlabel('Freq/kHz')
    plt.ylabel('Amplitude')
    plt.title(title)


if __name__ == '__main__':
    # 参数设置
    framerate = 16000  # 帧速率
    num_samples = 2000  # 每个缓冲区的帧数
    channels = 1  # 声道
    sampwidth = 2  # 设置样本宽度
    time = 5  # 录制时长

    # 录制音频
    record_path = "record.wav"
    # my_record("record")

    # 采样
    origin_sr = 10000  # 采样频率
    data_origin, _ = librosa.load(record_path, sr=origin_sr)
    soundfile.write("origin.wav", data_origin, origin_sr)

    # 高频余弦噪声
    f_cos = 4000
    noise_cos = 0.5 * np.cos(f_cos * np.arange(0, time, 1 / origin_sr)).reshape(data_origin.shape)
    soundfile.write("add_cos_noise.wav", data_origin + noise_cos, origin_sr)
    # 高斯白噪声
    percent = 0.2
    noise_gauss = percent * np.random.rand(len(data_origin))
    soundfile.write("add_guass_noise.wav", data_origin + noise_gauss, origin_sr)

    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
    # 原始音频
    plt.subplot(321)
    plot_wave("origin.wav", "原始信号时域波形")
    plt.subplot(322)
    plot_freq(data_origin, "原始信号频谱特性")
    # 叠加高频余弦噪声
    plt.subplot(323)
    plot_wave("add_cos_noise.wav", "叠加高频余弦噪声时域波形")
    plt.subplot(324)
    plot_freq(data_origin + noise_cos, "叠加高频余弦噪声频谱特性")
    # 叠加高斯白噪声
    plt.subplot(325)
    plot_wave("add_guass_noise.wav", "叠加高斯白噪声时域波形")
    plt.subplot(326)
    plot_freq(data_origin + noise_gauss, "叠加高斯白噪声频谱特性")
    plt.tight_layout()
    plt.show()

    # 播放音频对比
    print("原始音频")
    playwav("origin.wav")
    print("叠加高频余弦噪声")
    playwav("add_cos_noise.wav")
    print("叠加高斯白噪声")
    playwav("add_guass_noise.wav")
