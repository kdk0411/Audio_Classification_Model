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
File N주(Curse of Dimensionality), Over Fitting 등의 문제가
    발생 하면서 이에 해결책으로 나타난 것이 Residual_Block입니다.
    Residual_Block은 일반적인 layer와는 다르게 Output에서 자기 자신을 더한다는 특징을 가지고 있습니다.
    이를 '<strong>잔차블록</strong>'이라고도 합니다. 잔차는 모델간의 입력과 출력간의 차이를 말합니다.
    이 잔차를 수식으로 표현하면 아래와 같습니다.
      R(x) = Output - Input = H(x) - x
      이를 정리하면
      H(x) = R(x) + x 로 표현할 수 있다.
    잔차 블록은 실제 출력인 H(x)를 학습하려고 합니다. 또한 아래 그림과 같이 x로부터 항등 연결이
    있기 떄문에 layer는 실제로 잔차 R(x)를 학습한다는 것을 알 수 있습니다.
  </pre>
  </div>
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
