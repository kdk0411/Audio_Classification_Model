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
이러한 계수는 음성의 주파수 특성을 나타내며, MFCC 벡터의 각 요소로 사용됩니다. 
        </li>
      </ol>
  </pre>
  </div>
</details>
  
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

