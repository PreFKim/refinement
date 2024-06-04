# Root ReFineNet

## Problem

Diffusion으로 모션을 생성하는 경우 모션은 잘 생성되나 Root Position이 정확하게 생성이 되지 않는 문제점이 있다.

실질적으로 Root Position에 대한 학습은 쉽게 이루어지지 않기 때문에 따로 Root Position을 추론하는 모델을 작성하는 것이 더 낫다는 것을 이해했다.

이를 통해 관절 사이의 움직임을 이해하도록 Transformer 구조를 활용하여 실질적인 Root Position을 예측할 수 있도록 하는 모델을 구성한다.

## Installation

1. 환경 설정
    
    ```
    conda env create -f environment.yaml
    ```

2. 데이터 셋  

    - [FineDance Git](https://github.com/li-ronghui/FineDance/)

    - [FineDance Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view)

    설치 데이터 셋 내부의 .npy 파일들을 ./data/motion에 옮김

    SMPLX_NEUTRAL.npz 파일을 ./render에 옮김

## Trouble shooting

- torch.cuda.is_avilable()을 해도 False가 뜨는 경우

    1. nvcc -V 로 cuda 버전 확인

    2. [파이토치 버전확인](https://pytorch.org/get-started/previous-versions/)에서 Cuda 버전과 환경에 맞는 명령어 선택
 
    3. conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

- render.py 관련

    1. libGL.so.1: cannot open shared object file: No such file or directory [참조](https://yuevelyne.tistory.com/entry/OpenCV-ImportError-libGLso1-cannot-open-shared-object-file-No-such-file-or-directory)
    
        ```
        apt-get update -y
        apt-get install -y libgl1-mesa-glx
        ```

    2. ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory [참조](https://yuevelyne.tistory.com/entry/OpenCV-ImportError-libGLso1-cannot-open-shared-object-file-No-such-file-or-directory)

        ```
        apt-get install -y libglib2.0-0
        ```
        설치 과정중에 6 Asia 선택 후 69 Seoul 선택 

    3. FileNotFoundError: [Errno 2] No such file or directory: 'Xvfb' [참조](https://stackoverflow.com/questions/32173839/easyprocess-easyprocesscheckinstallederror-cmd-xvfb-help-oserror-errno)

        ```
        apt-get install xvfb
        ```
    4. TypeError: must be real number, not NoneType 영상 생성 중에 발생 [참조](https://stackoverflow.com/questions/68032884/getting-typeerror-must-be-real-number-not-nonetype-whenever-trying-to-run-wr)
 
       ```
       pip install moviepy --upgrade
       ```
       업데이트 후 커널 재시작

## Directory

    Refine
    ├── data
    │   ├── motion
    │       ├── 001.npy
    │       ├── 002.npy
    │       ├── ...
    │       └── 211.npy
    ├── dataset
    │   └── ...
    ├── experiments
    │   └── ...
    ├── model
    │   └── ...
    ├── ...
    └── ...
