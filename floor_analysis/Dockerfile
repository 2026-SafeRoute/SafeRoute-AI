FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# GPU가 없는 EC2에서 CUDA 패키지를 받지 않도록
# PyTorch CPU 전용 저장소를 사용한다.
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# floor_analysis 실행에 필요한 파일만 복사한다.
COPY model.py ./
COPY run_real_floorplan_pipeline.py ./
COPY test_pipeline.py ./
COPY fastapi_service.py ./
COPY unet_finetuned.pth ./

EXPOSE 8000

CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]