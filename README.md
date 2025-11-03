# 개요

포세이돈은 온디바이스 TinyML을 활용, QRAME(Quantum-Resilient Adaptive Mesh E2EE) 프로토콜의 내부 보안 AI 모델입니다. 이 모델은 패킷 엔트로피(
entropy), 타이밍 변동, 양자 노이즈(quantum noise)를 분석하며, 양자 시뮬레이터(quantum simulator)와 실제 공격 데이터셋(datasets)을 기반으로 훈련(학습)됩니다. 이러한
기능에
따라 이상 패킷 발견 시 즉시 PQC(Post-Quantum Cryptography) 알고리즘의 보안 수준을 상향 조정하고 그룹 경고를 발송하는 역할을 합니다.

기술적으로 자세한 사항은 [INTRODUCTION](INTRODUCTION.md) 문서를 확인하세요.

# 사용 전 세팅

사용자의 데이터셋으로 포세이돈을 학습시키거나, 학습된 포세이돈을 사용하려면 우선 몇 가지의 설정을 수행해야 합니다. 다음 명령을 통해 포세이돈을 `clone` 할 수 있습니다.

```shell
$ git clone https://github.com/Quant-Off/poseidon.git
$ cd poseidon
```

그리고 `.env` 파일을 만들어 다음과 같이 정의하세요.

```dotenv
# 플롯 시각화에 사용되는 폰트 ttf 파일 경로
MATPLOTLIB_FONT_PATH=../assets/SUIT-Light.ttf
# 플롯 시각화에 사용되는 폰트 이름 (확장자 제거)
MATPLOTLIB_FONT_NAME=SUIT-Light

# 데이터셋 디렉토리 절대경로
DATASETS_DIR_PATH=/Library/Quant/Repository/projects/poseidon/datasets
# 원본 데이터셋이 저장된 디렉토리의 경로 
DATASETS_ORIGIN_PATH="${DATASETS_DIR_PATH}/origin"
# 리샘플링된 데이터셋이 저장될 디렉토리의 경로
DATASETS_RESAMPLED_PATH="${DATASETS_DIR_PATH}/resampled"
# poseidon/data/custom_datasets.py 의 함수를 통해 만들어진 임의의 데이터셋이 저장될 디렉토리의 경로
DATASETS_CUSTOM_PATH="${DATASETS_DIR_PATH}/custom"
```

저희 팀은 포세이돈을 학습시킬 때 몇 가지의 추가적인 설정을 수행했지만 아직 많이 불안정해 현재 버전에서 공개할 수 없습니다.

사용자가 보유한 데이터셋을 학습시키려는 경우, `.env` 파일을 통해 할당한 경로에 디렉토리를 생성하고 원본 데이터셋을 옮겨두세요.

> 참고: 포세이돈은 학습에 원본 데이터셋을 변경하지 않지만, 메모리에 데이터를 복제하기 때문에 `데이터셋 용량x2` 이상의 메모리가 권장됩니다. 사용에 주의하세요.

# 리샘플링 및 피처 엔지니어링 수행

`poseidon.prototype.procession` 모듈에는 `all_process()` 함수가 존재합니다. 해당 함수에 전달값을 (아직까지는) 수동으로 변경하여 사용하고자 하는 데이터셋의 형식을 결정하세요.
예를 들어, 원본 데이터셋이 `poseidon/data/custom_datasets.py` 파일 또는 사용자가 테스트하고자 하는 데이터셋인 경우 실행 조건문에 `is_test = True` 값을 전달할 수 있습니다.
오버샘플링된 데이터셋인 경우엔 `is_smote = True` 값을 전달하면 됩니다. 그리고 다음과 같이 모듈을 실행하세요.

> 포세이돈 학습에 사용된 데이터셋 피처를 확인하려면 [NetFlow_v3_Features.csv](NetFlow_v3_Features.csv)파일을 확인할 수 있습니다.

```shell
$ python -m poseidon.prototype.processing
```

그러면 내부적으로 k-최근접 이웃, SMOTE 오버샘플링 연산을 수행하고 리샘플링된 데이터셋을 **저장하지 않고** 학습 연산을 수행합니다.

> 리샘플링된 데이터셋을 저장하려면 `poseidon/prototype/dataset_resampler.py` 파일의 `resample_dataset()` 함수를 사용하세요. 안정화 작업을 진행하고 나서
> 이 작업을 간편히 수행할 수 있도록 수정하겠습니다.

만약 이미 리샘플링된 데이터셋을 가지고 있다면 피처 엔지니어링 도구를 사용할 수 있습니다. 이 도구는 `poseidon/prototype/feature_engineering.py` 파일에 내장되어 있으며,
다음 순서와 같이 작업을 수행할 수 있습니다.

1. `X`, `y` 분리 작업
2. 스케일링
3. 피처 분석 및 히스토그램 출력
4. 섀넌 엔트로피를 통한 패킷 분석 후 결과 피처 생성
5. 타이밍 변동 측정 연산 수행 후 결과 피처 생성
6. 양자 시뮬레이션 적용 후 결과 피처 생성

그러면 최종 데이터셋이 저장되고 바로 훈련을 진행하실 수 있습니다.

사용자가 보유한 데이터셋을 기반으로 학습시키기 위한 몇 k-최근접 이웃, SMOTE 오버샘플링 함수와 엔트로피, 타이밍 변동 측정 등의 수학 유틸리티를 내장하고 있습니다.
아직 충분한 안정화 및 최적화 작업을 거치지 않았기 때문에 사용에는 반드시 주의하셔야 하며, 데이터셋을 리샘플링한 경우 해당 데이터셋을 엄밀하게 판단하세요.

위와 같은 함수들을 자유 자재로 사용하시거나 몇 가지 효율적 작업을 거친 뒤 문제의 부분을 발견하신 경우 이슈를 적극적으로 열어주시면 곧바로 응답하겠습니다.

# 기여

아직 포세이돈은 (현재 단계에서) 기여 받을 수 없습니다. 그럼에도 불구하고 당신의 지원에 감사드립니다!
