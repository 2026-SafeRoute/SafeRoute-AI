# SafeRoute Raspberry Pi congestion observer

Raspberry Pi가 CCTV 영상의 사람 수(기본은 화면 전체, ROI를 설정하면 ROI 안)를 5 FPS로 추론하고, Spring BE 설정에 따라 관측값·혼잡 이벤트·JPEG 이미지를 전송한다. Pi는 `GridCell`, `MapEdge`, 경로 계산, 관리자 승인 처리를 하지 않는다.

## 현재 연동 계약

- 장치 코드(`CCTV_CODE`)는 BE에 등록된 CCTV 코드를 그대로 쓰고(예: `CCTV_122`), 각 장치의 전용 Bearer Token은 `DEVICE_AUTH_TOKEN`으로만 주입한다. 코드와 토큰의 일치 여부는 BE가 검증한다.
- `GET /api/v1/device/congestion-config?cctvCode=...`를 훈련 중 5초, 비활성 중 1초 간격으로 조회한다. 비활성 간격을 짧게 둬서 훈련 시작을 빨리 알아챈다.
- `trainingSessionId`는 BE가 준 UUID를 그대로 사용한다. Pi가 세션 ID를 생성하지 않는다.
- `trainingActive=false`이면 추론, 관측값, 이벤트, 이미지 인코딩·업로드, Presigned URL 요청을 중단한다.
- 녹화 영상 입력은 `trainingActive=true`가 될 때까지 재생을 멈춰 두고, 훈련이 시작되면 멈춘 위치(처음이면 첫 프레임)부터 재생한다.
- `configVersion` 또는 세션/활성 상태가 바뀌면 집계 창, 추론 FPS, 임계값과 이벤트 설정을 즉시 적용한다.
- 밀도는 `headcount / monitoredAreaM2`로 계산하고 단계 임계값은 BE 응답만 사용한다.
- 모니터링 이미지와 같은 Snapshot의 집계 인원은 `frameHeadcount`로 보내며, 이미지 오버레이의 `headcount`와 항상 일치한다.
- 혼잡 진입/상승은 기본 3프레임, 정상 복귀는 5프레임 연속 조건이며 단계 상승은 cooldown과 무관하게 즉시 보낸다.
- 모든 시간 필드는 Unix timestamp 밀리초다.
- 훈련이 활성화된 직후 첫 프레임은 집계 구간을 기다리지 않고 초기 스냅샷으로 바로 보낸다. 초기 스냅샷은 `windowStart = windowEnd = capturedAt`, `sampleCount = 1`인 관측값이며, 이후에는 정기 집계 관측값이 이어진다.
- 혼잡 이벤트는 전용 단일 워커가 순서대로 보내고, 모니터링 스냅샷은 별도 워커 풀(`DELIVERY_MONITORING_WORKERS`, 기본 2)이 병렬로 보낸다. 모니터링 관측값은 도착 순서가 바뀔 수 있고, BE가 `capturedAt` 기준으로 최신 상태만 반영한다.
- 모니터링 스냅샷 JPEG는 최대 폭 `MONITORING_IMAGE_MAX_WIDTH`(기본 960px), 품질 `MONITORING_JPEG_QUALITY`(기본 70)로 줄여서 업로드한다. 혼잡 이벤트 이미지는 증거용이라 원본 해상도를 유지한다.
- 녹화 영상(`FILE_REALTIME=true`) 처리가 재생 속도보다 밀리면 오래된 프레임은 디코딩하지 않고 건너뛰어 최신 프레임부터 처리한다.

5초 관측값의 `avgHeadcount`는 정확도를 위해 실수로 보낸다. `peakHeadcount`와 `sampleCount`는 정수다. 이미지 업로드가 실패해도 `monitoringImageKey: null`로 관측값은 전송한다.

```json
{
  "eventId": "observation-uuid",
  "trainingSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "cctvCode": "CCTV_001",
  "avgHeadcount": 4.75,
  "peakHeadcount": 8,
  "frameHeadcount": 7,
  "sampleCount": 25,
  "windowStart": 1786500000000,
  "windowEnd": 1786500005000,
  "capturedAt": 1786500005000,
  "monitoringImageKey": null,
  "configVersion": 1
}
```

`avgHeadcount`와 `peakHeadcount`는 5초 관측 구간의 평균과 최대값이고,
`frameHeadcount`는 `monitoringImageKey`가 가리키는 단일 이미지의 인원수다.
BE는 상세 모니터링 프레임의 인원수와 밀집도를 `frameHeadcount` 기준으로 제공하고,
현재 혼잡 상태와 경로 계산에는 기존 구간 평균을 계속 사용할 수 있다.

혼잡 이벤트에는 `edgeId`와 `eventImageKey`를 넣지 않는다. 이벤트 POST와 이미지 업로드를 병렬 처리한 뒤, 둘 다 성공하면 `PATCH /api/v1/device/congestion-events/{eventId}/image`로 BE가 발급한 `objectKey`를 연결한다.

## 설치 및 실행

Python 3.9 이상 가상환경에서 다음을 실행한다.

```powershell
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

`.env.example`을 `.env`로 복사하고 실제 값으로 바꾼다. 토큰과 RTSP 비밀번호는 로그에 출력하지 않으며 `.env`는 커밋하지 않는다.

```powershell
$env:CCTV_CODE='CCTV_001'
$env:DEVICE_AUTH_TOKEN='{CCTV_001 전용 토큰}'
$env:SAFEROUTE_SERVER_BASE_URL='https://{BE_HOST}'
$env:VIDEO_SOURCE='{VIDEO_FILE_OR_RTSP_URL}'
python -m raspberry_pi_congestion.main file
# 또는
python -m raspberry_pi_congestion.main rtsp
```

### 여러 CCTV 동시 실행

공용 `.env`에는 `SAFEROUTE_SERVER_BASE_URL`, 모델 설정처럼 모든 CCTV가 공유하는
값을 둔다. `device-configs/CCTV.env.example`을 복사해 CCTV마다 별도 프로필을 만든다.

```text
device-configs/
  CCTV_001.env
  CCTV_002.env
```

각 프로필에는 장치별 값을 넣는다. DB에 저장된 해시가 아니라 CCTV 등록 시 발급된
원본 토큰을 사용해야 한다.

```dotenv
CCTV_CODE=CCTV_001
DEVICE_AUTH_TOKEN={CCTV_001 원본 토큰}
VIDEO_SOURCE={CCTV_001_VIDEO_FILE}
```

아래 명령 하나로 `device-configs/*.env`의 CCTV를 각각 독립 프로세스로 실행한다.

```powershell
python -m raspberry_pi_congestion.multi_main file
```

특정 프로필만 실행하려면 `--device-env`를 반복 지정한다.

```powershell
python -m raspberry_pi_congestion.multi_main file `
  --device-env device-configs/CCTV_001.env `
  --device-env device-configs/CCTV_002.env
```

로그에는 `[CCTV_001]`처럼 장치 코드가 표시된다. `OFFLINE_QUEUE_DB_PATH`를
생략하면 CCTV별 SQLite 파일을 자동으로 사용하고, 같은 큐 경로를 중복 지정하면
실행 전에 오류로 막는다. `Ctrl+C`를 누르면 실행 중인 CCTV 프로세스를 모두 종료한다.

## Hailo NPU 실행

현재 Hailo backend는 HailoRT 4.x 기반 Hailo-8/Hailo-8L과 다음 계약의
객체검출 HEF를 지원한다.

- 단일 NHWC RGB 입력(`uint8`)
- `HAILO_NMS_BY_CLASS` 형식의 단일 출력
- COCO class 순서(`person` class id `0`)
- NMS row: `[ymin, xmin, ymax, xmax, score]` 정규화 좌표

일반 `yolov8n.pt`는 Hailo에서 실행할 수 없다. 장치 아키텍처와 설치된
HailoRT 버전에 맞는 `.hef`를 사용해야 한다. Hailo-10H는 현재 지원 범위가
아니며 별도의 HailoRT 5.x 어댑터 검증이 필요하다.

AI Kit 또는 Hailo-8/8L AI HAT+에서 Hailo 패키지가 아직 없다면 Raspberry Pi
OS의 공식 패키지를 설치하고 재부팅한다.

```bash
sudo apt update
sudo apt install -y dkms hailo-all
sudo reboot
```

`multi_main`은 CCTV마다 별도 프로세스를 띄우고, 각 프로세스는 HailoRT multi-process
service를 거쳐 Hailo 장치 하나를 나눠 쓴다. `hailo-all`이 설치하는 `hailort.service`가
실행 중이어야 하며, 꺼져 있으면 CCTV가 한 대여도 Hailo 초기화에 실패한다.

```bash
systemctl status hailort --no-pager
```

장치와 모델을 확인한다.

```bash
hailortcli fw-control identify
dmesg | grep -i hailo
find /usr/share/hailo-models -type f -name '*.hef' 2>/dev/null
hailortcli run /실제/모델/경로/model.hef
```

APT로 설치한 `hailo_platform`이 기존 가상환경에서 보이지 않으면 시스템
site-packages를 사용하는 가상환경을 만든 뒤 Pi 의존성을 다시 설치한다.

```bash
python3 -m venv --system-site-packages .venv-hailo
source .venv-hailo/bin/activate
pip install -r requirements-pi.txt
pip install -e . --no-deps
python -c "import hailo_platform; print('hailo_platform import 성공')"
```

서버 전송 없이 CCTV별로 먼저 검증한다.

```dotenv
RUN_MODE=dry-run
CCTV_CODE=CCTV_001
VIDEO_SOURCE=rtsp://{USER}:{PASSWORD}@{CCTV_IP}:554/{STREAM_PATH}
DETECTOR_BACKEND=hailo
MODEL_PATH=/실제/모델/경로/model.hef
DETECTOR_CONF_THRESHOLD=0.4
TARGET_INFERENCE_FPS=5
SHOW_PREVIEW=false
```

```bash
python -m raspberry_pi_congestion.main dry-run
```

5초 관측 로그의 `sampleCount`가 약 25이면 유효 처리 속도가 약 5 FPS다.
`CCTV_001`, `CCTV_002`를 각각 검증한 뒤 `RUN_MODE=rtsp`로 백엔드 통합
테스트를 수행한다. RTSP URL, 카메라 비밀번호, 장치 토큰은 커밋하지 않는다.

개발 PC에서 추론 화면을 확인하려면 `SHOW_PREVIEW=true`를 설정한다. 초록색 박스는 집계된
사람이다. ROI를 설정한 경우 노란색 선이 ROI, 주황색 박스가 ROI 밖 사람이다. 창에서 `Q` 또는
`Esc`를 누르면 파이프라인과 미리보기 창이 함께 종료된다. 기본값은 `false`이다.
전송용 모니터링·이벤트 JPEG에도 같은 박스와 인원수가 표시된다.

기본값은 ROI 없이 화면 전체의 사람을 집계한다. 특정 영역만 집계하려면 `setup-roi`로 ROI를
CCTV별 파일로 저장한 뒤 `ROI_CONFIG_PATH`에 그 경로를 지정한다. `setup-roi`는
`ROI_CONFIG_PATH`가 없으면 `./config/roi/{CCTV_CODE}.json`에 저장한다.

```powershell
python -m raspberry_pi_congestion.main setup-roi
```

서버 없이 검출/집계를 확인하려면 `dry-run` 또는 `test` 모드를 사용한다. 이 두 모드만 로컬 기본 설정을 사용하며 운영 모드는 반드시 BE 설정을 조회한다.

## 상시 실행(systemd)

운영 Pi는 rtsp 모드를 계속 켜 두고, BE의 `trainingActive`에 따라 추론을 시작하고 멈춘다.
`deploy/raspberry-pi-congestion.service`의 경로를 실제 클론 위치와 가상환경에 맞게 고친 뒤 등록한다.

```bash
sudo cp deploy/raspberry-pi-congestion.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now raspberry-pi-congestion
journalctl -u raspberry-pi-congestion -f
```

RTSP 재연결이 `RTSP_MAX_RECONNECTS`번 연속 실패하면 프로세스는 종료 코드 1로 끝나고,
`multi_main`은 나머지 CCTV 프로세스도 정리한 뒤 같은 코드로 종료한다. 서비스는
`Restart=always`라서 `RestartSec` 뒤 전체를 다시 띄운다. 직접 실행 중인 프로세스와 서비스를
동시에 켜면 Hailo 장치를 두고 충돌하므로 테스트할 때는 서비스를 멈춘다.

## 재시도와 로컬 큐

200/201을 포함한 모든 2xx는 성공이다. 5xx, timeout, 429는 동일한 `eventId`, 세션, 시간, 버전, payload로 제한 재시도한 뒤 SQLite 큐에 보관한다. 400/401/403 같은 영구 오류는 큐에 넣지 않는다. 이벤트 이미지 연결의 404/409는 이벤트 생성 순서를 고려해 재시도한다. Presigned URL은 큐에 저장하거나 재사용하지 않는다.

SQLite 큐는 관측값, 혼잡 이벤트, 이벤트 이미지 연결 작업을 구분해서 저장한다. `trainingActive=false`일 때 종료 세션의 기존 큐를 자동 전송하지 않으며, 최종 처리 정책은 BE와 별도 합의가 필요하다.

## 주요 구조

```text
src/raspberry_pi_congestion/
  api_client.py          설정/관측/이벤트/Presigned URL/이미지 연결 API
  app.py                 polling, 추론, 병렬 이벤트·이미지 처리, 큐 재전송
  event_detector.py      연속 프레임과 cooldown 상태 머신
  models.py              최종 BE DTO와 설정 모델
  offline_queue.py       작업 유형별 SQLite 실패 큐
  window_aggregator.py   실수 평균·최대·성공 프레임 수 집계
```

Hailo 코드는 장치가 없는 CI에서 mock으로 검증한다. 실제 호환성, 처리 속도와
장시간 안정성은 대상 라즈베리파이·HailoRT·HEF 조합에서 별도로 검증해야 한다.
