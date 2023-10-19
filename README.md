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
  
  all_wav_paths = list(Data_dir.glob('*.wav'))
  for wav_path in all_wav_paths:
  
      y, sr = librosa.load(wav_path, sr=None)

      mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
      MFCC_list.append(mfcc)

      plt.figure(figsize=(10, 4))
      librosa.display.specshow(mfcc, x_axis='time')
      plt.colorbar()
      plt.title(f"MFCC - {wav_path.name}")
      plt.tight_layout()
      plt.show()
  print(f"Total MFCC_Auodio_Data Count : ", {len(MFCC_list)})
  ```
