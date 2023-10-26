# Audio_Classification_Model
<h3>목차</h3>
<ul>
  <li>프로젝트 설명</li>
  <li>데이터셋 설명</li>
  <li>MFCC</li>
  <li>Mel-Spectrogram</li>
  <li>Residual_Block</li>
  <li>결과 및 보고</li>
</ul>
<h1>프로젝트 설명</h1>
<p>
  신생아의 상태를 분류하는것을 목표로 하는 프로젝트입니다.
  해당 프로젝트에서는 배고픔, 배변 그외 3가지로 분류하였습니다.
  모델에서는 배고픔 그외 2가지로 분류하는것을 목표로 하였습니다.
  신생아의 상태별 울음소리를 분류하는 모델입니다. 해당 모델은 CNN을 기반으로
  하여 Residual_Block추가하여 만들었습니다.
</p>

<h1>데이터셋 설명</h1>
<p>
  데이터는 Kaggle에 존재하는 데이터셋을 사용하였습니다.
  데이터셋의 신빙성이 부족하여 데이터셋의 선정에있어서
  보다 더 명확한 데이터만을 채택하였습니다.
  음성 데아터의 구성은 배고픔 음성데이터와 그외의 4가지의
  음성 데이터로 구성되어 있으며 세부적으로는 성별, 개월수로 구분되어 있습니다.
</p>
<p>배고픔 음성 데이터 277개 그외 176개를 사용하여 모델을 학습 및 테스트하였습니다. 학습 데이터로는 254개를 테스트 데이터로는 23개를 사용하였습니다.</p>
<p>아래 코드는 그 중 일부를 가져와 음성 데이터의 기본 정보를 출력하였습니다.</p>

```Python
import os
import pathlib
import librosa
import numpy as np

# 데이터 디렉토리 경로 설정
data_directory = '/Classificant_Audio_data_test'
data_dir = pathlib.Path(data_directory)

# WAV 파일 경로 가져오기
audio_paths = sorted(list(data_dir.glob('*.wav')))

# 오디오 정보 수집
audio_info = []

for wav_path in audio_paths:
        y, sr = librosa.load(wav_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)  # 오디오 길이

        info = f"File Name: {wav_path.name} | Shape: {y.shape} | Sampling Rate: {sr} | Duration: {duration} seconds"
        audio_info.append(info)
# 결과 출력
for info in audio_info:
    print(info)

print(f"Total Output Count: {len(audio_info)}")
```
<pre>결과
File Name: awake_42.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: awake_43.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: awake_44.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: diaper_42.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: diaper_43.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: diaper_44.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: hug_42.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: hug_43.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: hug_44.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: hungry-f-04-07.wav | Shape: (48000,) | Sampling Rate: 8000 | Duration: 6.0 seconds
File Name: hungry-f-04-08.wav | Shape: (48000,) | Sampling Rate: 8000 | Duration: 6.0 seconds
File Name: hungry-f-04-09.wav | Shape: (48000,) | Sampling Rate: 8000 | Duration: 6.0 seconds
File Name: sleepy_42.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: sleepy_43.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: sleepy_44.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: uncom_42.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: uncom_43.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
File Name: uncom_44.wav | Shape: (96000,) | Sampling Rate: 16000 | Duration: 6.0 seconds
Total Output Count: 18
</pre>

<h1>MFCC</h1>
<details>
  <summary>MFCC</summary>
  <div markdown="1">
    <pre>
    MFCC (Mel-frequency cepstral coefficients)은 음성 및 오디오 신호 처리 분야에서 
    중요한 특징 추출 기술 중 하나입니다. MFCC는 음성 데이터의 중요한 주파수 및 스펙
    트럼 특성을 다룹니다. 이는 음성 인식, 음성 분류등에 사용하는데 중요한 역할을 합니다.
      <ol>MFCC의 주요특징
        <li>주파수 스케일 변환(Mel-scale)
 MFCC는 주파수 영억을 Mel-scale로 변환합니다. 이것은 인간의 청각
시스템의 특성을 사용하는데 필요합니다. Mel-scale은 낮은 주파수에
민감하며 높은 주파수에는 둔감하다는 특징을 가집니다.
        </li><li>로그 스케일 변환
 Mel-scale로 변환한 주파수 스펙트럼에 로그 스케일 변환을 적용합니다.
이는 주파수 스펙트럼의 크기를 줄이며 음성의 다양한 주파수 구성 요소를 강조 할 수 있습니다.
        </li><li>MFCC 특징벡터 추출
 로그 스케일 스펙트럼에서 MFCC 특징 벡터를 추출합니다.
이러한 계수는 음성의 주파수 특성을 나타내며, MFCC 벡터의 각 요소로 사용됩니다.</li></ol></pre></div>
</details>
<p>아래 코드는 MFCC를 이용한 2가지의 음성 데이터의 시각화 코드입니다.</p>

  ```Python
  import os
  import pathlib
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt

  Dataset_Path = '/Audio'
  Data_dir = pathlib.Path(Dataset_Path)
  MFCC_list = []
  # 모든 .wav 파일 경로를 가져옵니다.
  all_wav_paths = list(Data_dir.glob('*.wav'))
  # wav파일에 대한 MFCC 추출 및 시각화를 진행합니다.
  for wav_path in all_wav_paths:
  
      y, sr = librosa.load(wav_path, sr=None)
      # MFCC 추출 (40의 계수를 사용)
      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
      # MFCC 리스트에 추가
      MFCC_list.append(mfcc)

      # MFCC 시각화
      plt.figure(figsize=(10, 4))
      librosa.display.specshow(mfcc, x_axis='time')
      plt.colorbar()
      plt.title(f"MFCC - {wav_path.name}")
      plt.tight_layout()
      plt.show()
  print(f"Total MFCC_Auodio_Data Count : ", {len(MFCC_list)})
  ```
![MFCC_Hug_f_04](https://github.com/kdk0411/Audio_Classification_Model/assets/99461483/80e71de2-ded5-4e71-b0ab-aed1b1a3503c)
![MFCC_Hug_m_04](https://github.com/kdk0411/Audio_Classification_Model/assets/99461483/03833127-91ef-4521-8749-0a0d928c8c28)

<h1>Mel-Spectrogram</h1>
<details>
  <summary>Mel-Spectrogram</summary>
  <div markdown="1">
    <pre>
    Mel-Spectrogram은 오디오 신호의 주파수 내용을 시간에 따라 표현한 그래프인 Spectrogram을
    Mel 스케일(Mel Scale)로 변환한 것입니다. 결과적으로 특정 시간에 오디오 신호의 주파수 내용을
    Mel 스케일로 표현한 그래프라고 할 수 있습니다.
      <ul><li><strong>Mel 스케일(Mel Scale)</strong>
Mel 스케일은 인간의 청각 특성에 근거한 주파수 스케일입니다. Mel 스케일은
주파수 간의 간격이 Spectrogram과 달리 인간 청각 시스템의 높은 감도를
반영하도록 조정되었기 때문에 음성 신호 내의 주파수 성분을 더욱 자연스럽게
표현할 수 있습니다. 이는 중요한 부분을 강조하고 불필요한 부분을 무시하며
인간의 청각 시스템과 비슷하게 만듭니다.</li></ul>
  </pre>
  </div>
</details>
<p>아래 코드는 Mel-Spcetorgram을 이용하여 2가지의 음성 데이터를 시각화한 코드입니다.</p>

```Python
import os
import pathlib
import librosa
import librosa.display
import matplotlib.pyplot as plt

Dataset_Path = '/Audio'
Data_dir = pathlib.Path(Dataset_Path)

# 모든 .wav 파일 경로를 가져옵니다.
all_wav_paths = list(Data_dir.glob('*.wav'))
# wav파일에 대한 Mel-Spectrogram 추출 및 시각화를 진행합니다.
for wav_path in all_wav_paths:
    y, sr = librosa.load(wav_path, sr=None)
    # Mel-Spcetrogram을 추출합니다.
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    # power_to_db를 이용하여 데시벨 단위로 변환합니다.
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Mel-Spectrogram을 시각화 합니다.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2)
    plt.colorbar()
    plt.title(f"Mel-Spectrogram - {wav_path.name}")
    plt.tight_layout()
    plt.show()
```
![Mel_Spectorgram_Hug_f_04](https://github.com/kdk0411/Audio_Classification_Model/assets/99461483/f02290af-b884-4155-80f6-f68beaab34a1)
![Mel_Spectorgram_Hug_m_04](https://github.com/kdk0411/Audio_Classification_Model/assets/99461483/4fc79c71-f961-4227-bdff-f15f9bd044c8)


<h1>Residaul_Block</h1>
<details>
  <summary>Residaul_Block</summary>
  <div markdown="1">
    <pre>
    Layer의 숫자를 늘리는 것이 모델성능을 무조건적으로 향상 시켜주지 않습니다.
    또한 기울기 손실(Gradient Vanising), 차원의 저주(Curse of Dimensionality), Over Fitting 등의 문제가
    발생 하면서 이에 해결책으로 나타난 것이 Residual_Block입니다.
    Residual_Block은 일반적인 layer와는 다르게 Output에서 자기 자신을 더한다는 특징을 가지고 있습니다.
    이를 '<strong>잔차블록</strong>'이라고도 합니다. 잔차는 모델간의 입력과 출력간의 차이를 말합니다.
    이 잔차를 수식으로 표현하면 아래와 같습니다.
      R(x) = Output - Input = H(x) - x
      이를 정리하면
      H(x) = R(x) + x 로 표현할 수 있다.
    잔차 블록은 실제 출력인 H(x)를 학습하려고 합니다. 또한 아래 그림과 같이 x로부터 항등 연결이
    있기 떄문에 layer는 실제로 잔차 R(x)를 학습한다는 것을 알 수 있습니다.</pre></div>
</details>

```Python
# Residual Block 함수생성
def residual_block(inputs, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])
    x = Activation('relu')(x)
    return x

from tensorflow.keras.layers import Input, concatenate, Conv2D, Reshape, BatchNormalization, Activation, Add, Flatten
from tensorflow.keras.layers import LeakyReLU
# MFCC 모델 생성
mfcc_input = Input(shape=X_train_mfcc_scaled.shape[1:])
mfcc_reshaped = Reshape((*X_train_mfcc_scaled.shape[1:], 1))(mfcc_input)
mfcc_model = Conv2D(32, kernel_size=(4, 4))(mfcc_reshaped)
mfcc_model = BatchNormalization()(mfcc_model)
mfcc_model = LeakyReLU()(mfcc_model)

# Mfcc Model에 Residual Block 추가
mfcc_model = residual_block(mfcc_model, 32, (4, 4))
mfcc_model = residual_block(mfcc_model, 32, (4, 4))

# Mel-Spectrogram 모델 생성
mel_spec_input = Input(shape=(X_train_mel_spec_scaled.shape[1], X_train_mel_spec_scaled.shape[2], 1))
mel_spec_model = Conv2D(32, kernel_size=(4, 4))(mel_spec_input)
mel_spec_model = BatchNormalization()(mel_spec_model)
mel_spec_model = LeakyReLU()(mel_spec_model)

# Mel-Spectrogram Model에 Residual Block 추가
mel_spec_model = residual_block(mel_spec_model, 32, (4, 4))
mel_spec_model = residual_block(mel_spec_model, 32, (4, 4))

mfcc_model_flatten = Flatten()(mfcc_model)
mel_spec_model_flatten = Flatten()(mel_spec_model)
combined = concatenate([mfcc_model_flatten, mel_spec_model_flatten])
```


## 결과 및 보고

  아래 코드를 사용하여 Flask 에서 실행 하였습니다.

```python
import pandas as pd
import numpy as np
import librosa
import os
import pathlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score


class AudioClassifier:
    def __init__(self, wav_path, csv_path):
        self.wav_path = wav_path
        self.csv_path = csv_path

    def process_data(self):
        X_mfcc = []
        X_mel_spec = []
        labels = []
        data_dir = pathlib.Path(self.wav_path)
        all_wav_paths = sorted(list(data_dir.glob('*.wav')))

        df = pd.read_csv(self.csv_path)
        cry_audio_file = df["Cry_Audio_File"]
        label = df["Label"]

        max_length = 188

        for wav_path_dir in all_wav_paths:
            file_name = os.path.basename(wav_path_dir)
            index = cry_audio_file[cry_audio_file == file_name].index[0]
            label_value = label[index]

            y, sr = librosa.load(wav_path_dir, sr=16000, duration=6)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            if mfcc.shape[1] > max_length:
                mfcc = mfcc[:, :max_length]

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
            if mel_spec.shape[1] > max_length:
                mel_spec = mel_spec[:, :max_length]

            X_mfcc.append(mfcc)
            X_mel_spec.append(mel_spec)
            labels.append(label_value)

        X_mfcc = np.array(X_mfcc)
        X_mel_spec = np.array(X_mel_spec)
        labels = np.array(labels)
        return X_mfcc, X_mel_spec, labels

    def preprocess_data(self):
        X_mfcc, X_mel_spec, labels = self.process_data()

        scaler_mfcc = StandardScaler()
        scaler_mel_spec = StandardScaler()

        X_mfcc_scaled = scaler_mfcc.fit_transform(X_mfcc.reshape(-1, X_mfcc.shape[-1])).reshape(X_mfcc.shape)
        X_mel_spec_scaled = scaler_mel_spec.fit_transform(X_mel_spec.reshape(-1, X_mel_spec.shape[-1])).reshape(X_mel_spec.shape)

        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)

        return X_mfcc_scaled, X_mel_spec_scaled, labels_encoded, label_encoder, num_classes

test_csv_path = "/content/drive/MyDrive/Baby_Sound/Hungry/Classificant_Audio_data_test/test_Audio_New.csv"
test_data_dir = "/content/drive/MyDrive/Baby_Sound/Hungry/Classificant_Audio_data_test"

test_classifier = AudioClassifier(test_data_dir, test_csv_path)
X_test_mfcc, X_test_mel_spec, y_test, _, _ = test_classifier.preprocess_data()

# 모델 불러오기
model = load_model('/content/drive/MyDrive/Audio_Classify_Model_0.2_97.h5')

# 모델 평가
loss, accuracy = model.evaluate([X_test_mfcc, X_test_mel_spec], y_test)

y_pred = model.predict([X_test_mfcc, X_test_mel_spec])
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print("F1 Score:", np.ceil(f1*1000)/1000)
print("Test Loss:", np.ceil(loss*1000)/1000)
print("Test Accuracy:", np.ceil(accuracy*10000)/100, "%")

```

```
  2/2 [==============================] - 1s 22ms/step - loss: 0.2583 - accuracy: 0.9714
  2/2 [==============================] - 0s 59ms/step
  F1 Score: 0.972
  Test Loss: 0.259
  Test Accuracy: 97.15 %
```

  위 코드와 같이 정확도, 손실, F1 Score를 측정하였습니다.
  정확도는 97.15%, F1 Score는 0.972 손실은 0.259로 측정 되었습니다.

  이는 실제로 테스트 했을떄도 같은 결과를 도출 하였습니다.
  실제는 코드로 만든 정적인 것과 다르게 주변 노이즈와 Rpi에서의 자체적인 노이즈
  Rpi를 사용 하였을때 저항에 의한 노이즈 등을 생각했을떄는 제가 준비한 음성 데이터 이외에는
  이보다 더 낮은 점수를 기록할 수 있습니다.

  프로젝트를 진행함에 있어서 여러가지의 문제점이 발견되었습니다.
  1. 데이터 불균형
  2. 데이터셋의 부족
  3. 녹음기기 노이즈
  대표적으로는 이렇게 3가지 였습니다.

  먼저 데이터 불균형의 경우는 배고픔, 그외 이렇게 이진분류를 진행 하는것으로 팀원들과 상의 후 결정하였습니다
  물론 그외의 다른 경우도 포함 한다면 가능은 하겠지만 정확도가 확연히 떨어질 것을 우려하여 결정하게 된 사항입니다.

  두 번째로 데이터셋의 부족입니다. 외국 기업에 연락하여 아이의 울음소리 데이터를 요청해보았지만
  법률상 아이의 울음소리를 제공하는것이 문제가 될 수 있다는 답변들을 받아 부족함을 느꼇습니다.
  이는 Kaggle에서 또다른 데이터셋을 발견하게 되어 조금이나마 추가하게 되었습니다.
  이외에 여러 아이 울음소리 데이터는 많았지만 모두 신빙성이 떨어졌습니다.
  아이의 울음소리가 맞는지 맞더라도 어떤 이유떄문에 우는것인지 알 수 없었기 떄문에 데이터셋의 부족을 느꼇습니다.

  마지막으로 녹음기기의 노이즈입니다. 아이 울음소리를 기준으로 모델을 학습시켰지만 분명히 모델은
  녹음 기기의 노이즈 또한 학습하였을 것입니다. 이를 노이즈 제거 알고리즘을 이용하여 해결하려 했지만
  긍정적인 변화를 도출해낼 수 없을것이라 판단하여 기존방식대로 진행 하였습니다.
  이유는 학습된 데이터셋과 테스트 데이터셋의 출처가 다르며 실제 녹음기기 또한 노이즈가 존재하기 떄문에
  특정 노이즈를 모델이 학습하지 못했을 것이라 판단했기 떄문입니다.

  프로젝트를 마치며 가장 아쉬웠던 부분이 존재합니다.
  
