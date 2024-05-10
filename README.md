# Root ReFineNet

## Pytorch Environment

torch.cuda.is_avilable()을 해도 False가 뜨는 경우

1. nvcc -V 로 cuda 버전 확인

2. [파이토치 버전확인](https://pytorch.org/get-started/previous-versions/)에서 Cuda 버전과 환경에 맞는 명령어 선택


## Installation

1. conda env create -f environment.yaml

2. python train.py