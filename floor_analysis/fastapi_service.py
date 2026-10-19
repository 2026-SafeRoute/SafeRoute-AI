"""
Safe Route AI 분석 서비스 (FastAPI)

노드 타입: is_door -> DOOR, room -> ROOM, corridor -> HALLWAY, stairs -> STAIR (EXIT/CUSTOM은 관리자가 지정)
좌표는 0~1 정규화, 거리는 요청의 real_width_m/real_height_m로 환산한 미터.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import math
import tempfile
from typing import List

from dotenv import load_dotenv

load_dotenv()

import boto3
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from run_real_floorplan_pipeline import analyze_floorplan
from model import UNet


app = FastAPI(title="Safe Route AI Service")

MODEL_PATH = os.environ.get("SAFEROUTE_MODEL_PATH", "unet_finetuned_v8.pth")
S3_BUCKET = os.environ.get("SAFEROUTE_S3_BUCKET")
DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

# unet(기본) / rule. 환경변수로 전환
ANALYSIS_MODE = os.environ.get("SAFEROUTE_ANALYSIS_MODE", "unet").strip().lower()
if ANALYSIS_MODE not in ("rule", "unet"):
    print(f"경고: SAFEROUTE_ANALYSIS_MODE 값이 올바르지 않아 'unet'을 사용합니다. (입력값: {ANALYSIS_MODE})")
    ANALYSIS_MODE = "unet"

_model = None
_s3_client = None


@app.on_event("startup")
def load_resources():
    global _model, _s3_client

    print(f"모델 로딩 중... ({MODEL_PATH}, device={DEVICE})")
    _model = UNet(in_channels=3, num_classes=6).to(DEVICE)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _model.eval()
    print("모델 로딩 완료")

    if not S3_BUCKET:
        print("경고: SAFEROUTE_S3_BUCKET 환경변수가 설정되지 않았습니다.")
    _s3_client = boto3.client("s3")


def download_image_from_s3(image_key: str) -> Image.Image:
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="SAFEROUTE_S3_BUCKET 환경변수가 설정되지 않았습니다.")
    buf = io.BytesIO()
    try:
        _s3_client.download_fileobj(S3_BUCKET, image_key, buf)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3에서 이미지를 찾을 수 없습니다 (key={image_key}): {e}")
    buf.seek(0)
    return to_rgb_on_white(Image.open(buf))


def to_rgb_on_white(image: Image.Image) -> Image.Image:
    """투명 배경은 흰색 위에 합성. 그냥 RGB로 바꾸면 투명 영역이 검정(벽)이 된다."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


class AnalyzeRequest(BaseModel):
    image_key: str = Field(..., description="S3 상의 도면 이미지 키(경로)")
    real_width_m: float = Field(..., gt=0, description="도면이 나타내는 층의 실제 가로 길이(m)")
    real_height_m: float = Field(..., gt=0, description="도면이 나타내는 층의 실제 세로 길이(m)")
    image_size: int = Field(512, description="분석용 내부 리사이즈 크기")


class NodeOut(BaseModel):
    temp_id: str
    code: str
    type: str
    x: float
    y: float


class EdgeOut(BaseModel):
    from_temp_id: str
    to_temp_id: str
    distance: float
    bidirectional: bool = True


class AnalyzeResponse(BaseModel):
    plan_width_px: int
    plan_height_px: int
    nodes: List[NodeOut]
    edges: List[EdgeOut]
    stats: dict


def map_space_type_to_node_type(is_door: bool, space_type: str) -> str:
    if is_door:
        return "DOOR"
    mapping = {"room": "ROOM", "corridor": "HALLWAY", "stairs": "STAIR"}
    return mapping.get(space_type, "HALLWAY")


@app.post("/analyze-floorplan", response_model=AnalyzeResponse)
def analyze_floorplan_endpoint(req: AnalyzeRequest):
    original_image = download_image_from_s3(req.image_key)
    plan_width_px, plan_height_px = original_image.size

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        original_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = analyze_floorplan(
            tmp_path, model=_model, device=DEVICE, image_size=req.image_size, mode=ANALYSIS_MODE
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"도면 분석 중 오류: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    G = result["G"]
    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=422, detail="도면에서 노드를 하나도 찾지 못했습니다.")

    width_scale = req.real_width_m / req.image_size
    height_scale = req.real_height_m / req.image_size

    type_counters: dict = {}
    nodes_out = []
    for node_key, data in G.nodes(data=True):
        node_type = map_space_type_to_node_type(
            data.get("is_door", False), data.get("space_type", "corridor")
        )
        type_counters[node_type] = type_counters.get(node_type, 0) + 1
        code = f"{node_type.lower()}_{type_counters[node_type]}"

        nodes_out.append(
            NodeOut(
                temp_id=str(node_key),
                code=code,
                type=node_type,
                x=round(data["col"] / req.image_size, 4),
                y=round(data["row"] / req.image_size, 4),
            )
        )

    edges_out = []
    for u, v, edata in G.edges(data=True):
        dr = G.nodes[u]["row"] - G.nodes[v]["row"]
        dc = G.nodes[u]["col"] - G.nodes[v]["col"]
        distance_m = math.sqrt((dc * width_scale) ** 2 + (dr * height_scale) ** 2)
        if round(distance_m, 3) <= 0:  # BE는 0 이하 거리를 거부함
            distance_m = 0.01
        edges_out.append(
            EdgeOut(
                from_temp_id=str(u),
                to_temp_id=str(v),
                distance=round(distance_m, 3),
                bidirectional=True,
            )
        )

    return AnalyzeResponse(
        plan_width_px=plan_width_px, plan_height_px=plan_height_px,
        nodes=nodes_out, edges=edges_out, stats=result["stats"],
    )


@app.get("/health")
def health():
    s3_ok = None
    if S3_BUCKET:
        try:
            _s3_client.head_bucket(Bucket=S3_BUCKET)
            s3_ok = True
        except Exception:
            s3_ok = False

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": _model is not None,
        "analysis_mode": ANALYSIS_MODE,
        "bucket": S3_BUCKET,
        "bucket_accessible": s3_ok,
    }


# 개발용: S3 없이 로컬 파일로 테스트 (ENABLE_LOCAL_TEST=true)
if os.environ.get("ENABLE_LOCAL_TEST", "false").lower() == "true":

    class AnalyzeLocalRequest(BaseModel):
        local_image_path: str = Field(..., description="로컬 도면 파일 경로")
        real_width_m: float = Field(..., gt=0)
        real_height_m: float = Field(..., gt=0)
        image_size: int = Field(512)

    @app.post("/analyze-floorplan-local", response_model=AnalyzeResponse)
    def analyze_floorplan_local_endpoint(req: AnalyzeLocalRequest):
        if not os.path.exists(req.local_image_path):
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {req.local_image_path}")

        original_image = Image.open(req.local_image_path).convert("RGB")
        plan_width_px, plan_height_px = original_image.size

        try:
            result = analyze_floorplan(
                req.local_image_path, model=_model, device=DEVICE, image_size=req.image_size,
                mode=ANALYSIS_MODE,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"도면 분석 중 오류: {e}")

        G = result["G"]
        if G.number_of_nodes() == 0:
            raise HTTPException(status_code=422, detail="도면에서 노드를 하나도 찾지 못했습니다.")

        width_scale = req.real_width_m / req.image_size
        height_scale = req.real_height_m / req.image_size

        type_counters: dict = {}
        nodes_out = []
        for node_key, data in G.nodes(data=True):
            node_type = map_space_type_to_node_type(
                data.get("is_door", False), data.get("space_type", "corridor")
            )
            type_counters[node_type] = type_counters.get(node_type, 0) + 1
            code = f"{node_type.lower()}_{type_counters[node_type]}"
            nodes_out.append(NodeOut(
                temp_id=str(node_key), code=code, type=node_type,
                x=round(data["col"] / req.image_size, 4),
                y=round(data["row"] / req.image_size, 4),
            ))

        edges_out = []
        for u, v, edata in G.edges(data=True):
            dr = G.nodes[u]["row"] - G.nodes[v]["row"]
            dc = G.nodes[u]["col"] - G.nodes[v]["col"]
            distance_m = math.sqrt((dc * width_scale) ** 2 + (dr * height_scale) ** 2)
            if round(distance_m, 3) <= 0:
                distance_m = 0.01
            edges_out.append(EdgeOut(
                from_temp_id=str(u), to_temp_id=str(v),
                distance=round(distance_m, 3), bidirectional=True,
            ))

        return AnalyzeResponse(
            plan_width_px=plan_width_px, plan_height_px=plan_height_px,
            nodes=nodes_out, edges=edges_out, stats=result["stats"],
        )
