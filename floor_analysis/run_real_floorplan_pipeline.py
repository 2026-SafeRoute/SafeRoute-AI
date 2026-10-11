"""
파인튜닝된 모델(unet_finetuned.pth)의 예측 결과를 그래프로 변환하는
통합 테스트 스크립트 (하이브리드 노드 방식).

노드 생성 방식:
  - 방(room): 방 하나당 노드 1개만 생성 (벽에서 가장 먼 지점 = distance transform
    최대 지점). 단순 기하학적 중심이 아니라서 ㄱ자/ㄷ자 방이어도 실제로
    걸을 수 있는 위치를 보장함.
  - 복도(corridor)/계단(stairs)/문(door): 기존 스켈레톤 방식 그대로 유지.
    복도를 따라 스켈레톤(중심선)을 뽑고, 분기점/끝점마다 노드를 만듦.
    문은 스켈레톤 노드 중 문 근처에 있는 노드에 is_door=True 속성으로 표시됨
    (문 전용 노드를 따로 만들지 않음 - 원래 방식과 동일).
  - 방 노드와 복도 그래프는, 각 문이 어느 방과 맞닿아 있는지를 계산해서
    "방 중앙 노드 <-> 그 문에서 가장 가까운 복도 스켈레톤 노드"로 연결함.

- 출구(exit)는 지정하지 않으며, Dijkstra 대피경로 계산도 하지 않음.

세그멘테이션 후처리:
  - 문(door) 예측 노이즈 제거 + 과대 예측된 문 덩어리 축소
  - 얇거나 끊긴 벽 보강
  - 복도(corridor)와 계단(stairs)을 space_type 속성으로 구분

사전 준비:
  - 이 파일을 finetune_local_floorplans.py, test_pipeline.py,
    safe_route_cubicasa_training_pycharm.py 와 같은 폴더에 둘 것
  - unet_finetuned.pth 가 같은 폴더에 있을 것

실행:
    python run_real_floorplan_pipeline.py --image_path finetune_data/images/5.png
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import UNet
import test_pipeline as tp  # label_rooms, clean_and_skeletonize, skeleton_to_graph 등 재사용

try:
    from db_export import export_graph_to_postgres
    DB_EXPORT_AVAILABLE = True
except ImportError:
    DB_EXPORT_AVAILABLE = False


# 파인튜닝 모델의 6클래스 정의 (finetune_local_floorplans.py와 동일해야 함)
FT_WALL, FT_ROOM, FT_CORRIDOR, FT_DOOR, FT_WINDOW, FT_STAIRS = range(6)
SPACE_TYPE_NAMES = {0: "none", 1: "room", 2: "corridor", 3: "stairs"}

# 6클래스 -> test_pipeline.py의 3클래스(WALL/FLOOR/DOOR) 매핑
CLASS_REMAP = {
    FT_WALL: tp.WALL,
    FT_ROOM: tp.FLOOR,
    FT_CORRIDOR: tp.FLOOR,
    FT_DOOR: tp.DOOR,
    FT_WINDOW: tp.WALL,
    FT_STAIRS: tp.FLOOR,
}


def predict_mask_6class(model, image_path, device, image_size=512):
    model.eval()
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    img_arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return pred_mask  # 값 0~5


def remap_to_pipeline_classes(pred_mask_6class):
    remapped = np.zeros_like(pred_mask_6class, dtype=np.uint8)
    for src_cls, dst_cls in CLASS_REMAP.items():
        remapped[pred_mask_6class == src_cls] = dst_cls
    return remapped


def reinforce_walls(mask, iterations=1):
    """
    예측된 벽이 군데군데 끊겨 있으면 원래는 다른 방인 두 공간이
    하나로 합쳐지는 문제가 생김 (벽이 얇게 학습된 경우 흔함).
    벽을 room 쪽으로만 살짝 팽창시켜서 끊긴 부분을 이어줌.
    door는 침범하지 않도록 함 (문까지 막아버리면 그래프가 끊어짐).
    """
    if iterations <= 0:
        return mask
    wall_mask = (mask == tp.WALL)
    dilated_wall = ndimage.binary_dilation(wall_mask, iterations=iterations)
    overwrite = dilated_wall & (mask == tp.FLOOR)  # FLOOR 쪽으로만 확장, DOOR는 보존
    reinforced = mask.copy()
    reinforced[overwrite] = tp.WALL
    return reinforced


def build_space_type_map(pred_mask_6class):
    """room=1, corridor=2, stairs=3 (그 외 0)을 담은 맵."""
    space_type = np.zeros_like(pred_mask_6class, dtype=np.uint8)
    space_type[pred_mask_6class == FT_ROOM] = 1
    space_type[pred_mask_6class == FT_CORRIDOR] = 2
    space_type[pred_mask_6class == FT_STAIRS] = 3
    return space_type


def clean_door_noise(mask, min_size=25):
    """실제 문보다 훨씬 작은 문 조각(예측 노이즈)을 찾아서 FLOOR로 되돌림."""
    door_labels, num_doors = ndimage.label(mask == tp.DOOR)
    if num_doors == 0:
        return mask, 0, 0

    sizes = ndimage.sum(mask == tp.DOOR, door_labels, index=range(1, num_doors + 1))
    cleaned = mask.copy()
    removed = 0
    for door_id, size in enumerate(sizes, start=1):
        if size < min_size:
            cleaned[door_labels == door_id] = tp.FLOOR
            removed += 1

    return cleaned, num_doors, removed


def shrink_oversized_doors(mask, max_area=200, erosion_iterations=2):
    """실제 문 크기보다 훨씬 큰 문 덩어리를 침식시켜서 크기를 줄임."""
    door_labels, num_doors = ndimage.label(mask == tp.DOOR)
    if num_doors == 0:
        return mask, 0

    shrunk = mask.copy()
    num_shrunk = 0
    for door_id in range(1, num_doors + 1):
        component = (door_labels == door_id)
        area = int(component.sum())
        if area <= max_area:
            continue

        eroded = ndimage.binary_erosion(component, iterations=erosion_iterations)
        if eroded.sum() == 0:
            eroded = ndimage.binary_erosion(component, iterations=1)

        removed_part = component & ~eroded
        shrunk[removed_part] = tp.FLOOR
        num_shrunk += 1

    return shrunk, num_shrunk


def densify_long_edges(G, max_edge_length=40):
    """
    Graph Simplification으로 직선 구간의 중간 노드가 다 사라져서, 긴 복도가
    끝점(혹은 분기점) 노드 하나로만 표현되는 문제를 완화.
    엣지 길이가 max_edge_length를 넘으면, 그 사이에 일정 간격으로
    중간 노드(waypoint)를 새로 끼워 넣음.
    """
    edges_to_process = list(G.edges(data=True))
    for u, v, data in edges_to_process:
        weight = data.get("weight", 0)
        if weight <= max_edge_length:
            continue

        n_segments = int(np.ceil(weight / max_edge_length))
        if n_segments <= 1:
            continue

        r1, c1 = G.nodes[u]["row"], G.nodes[u]["col"]
        r2, c2 = G.nodes[v]["row"], G.nodes[v]["col"]
        # 중간 노드의 space_type은 두 끝점 중 문이 아닌 쪽을 따라감 (대개 같은 복도)
        space_type = G.nodes[u].get("space_type")
        if space_type in (None, "door"):
            space_type = G.nodes[v].get("space_type", "corridor")

        G.remove_edge(u, v)
        seg_dist = weight / n_segments
        prev_node = u
        for i in range(1, n_segments):
            t = i / n_segments
            r, c = r1 + (r2 - r1) * t, c1 + (c2 - c1) * t
            new_id = f"{u}__{v}__wp{i}"
            G.add_node(new_id, row=r, col=c, room_id=None, space_type=space_type, is_door=False)
            G.add_edge(prev_node, new_id, weight=seg_dist)
            prev_node = new_id
        G.add_edge(prev_node, v, weight=seg_dist)

    return G


def merge_nearby_nodes(G, node_filter, merge_radius):
    """
    node_filter(data) -> bool 조건을 만족하는 노드들 중, merge_radius 안에
    가깝게 몰려있는 것들을 하나로 합침. 합쳐지는 노드들이 갖고 있던 엣지는
    전부 대표 노드로 재연결함 (대표 노드 위치는 병합된 노드들의 평균 좌표).
    """
    target_nodes = [n for n, d in G.nodes(data=True) if node_filter(d)]

    proximity = nx.Graph()
    proximity.add_nodes_from(target_nodes)
    for i in range(len(target_nodes)):
        for j in range(i + 1, len(target_nodes)):
            n1, n2 = target_nodes[i], target_nodes[j]
            d1, d2 = G.nodes[n1], G.nodes[n2]
            dist = ((d1["row"] - d2["row"]) ** 2 + (d1["col"] - d2["col"]) ** 2) ** 0.5
            if dist <= merge_radius:
                proximity.add_edge(n1, n2)

    merged_count = 0
    for cluster in nx.connected_components(proximity):
        cluster = list(cluster)
        if len(cluster) <= 1:
            continue

        rows = [G.nodes[n]["row"] for n in cluster]
        cols = [G.nodes[n]["col"] for n in cluster]
        rep = cluster[0]
        G.nodes[rep]["row"] = float(np.mean(rows))
        G.nodes[rep]["col"] = float(np.mean(cols))

        for other in cluster[1:]:
            for neighbor in list(G.neighbors(other)):
                if neighbor == rep or neighbor in cluster:
                    continue
                w = G[other][neighbor].get("weight", 1.0)
                if G.has_edge(rep, neighbor):
                    w = min(w, G[rep][neighbor]["weight"])
                G.add_edge(rep, neighbor, weight=w)
            G.remove_node(other)
            merged_count += 1

    return G, merged_count


def merge_nearby_door_nodes(G, merge_radius=15):
    """같은 문 하나에 스켈레톤 노드가 여러 개 몰리는 문제를 완화."""
    return merge_nearby_nodes(G, lambda d: d.get("is_door"), merge_radius)


def merge_nearby_stairs_nodes(G, merge_radius=20):
    """같은 계단 하나에 스켈레톤 노드가 여러 개 몰리는 문제를 완화."""
    return merge_nearby_nodes(G, lambda d: d.get("space_type") == "stairs", merge_radius)


def compute_door_connections(mask, room_labels):
    """각 문 조각이 어느 방/복도 두 공간을 연결하는지 계산 (출구는 지정 안 함)."""
    door_component_labels, num_doors = ndimage.label(mask == tp.DOOR)

    door_info = {}
    for door_id in range(1, num_doors + 1):
        door_pixels = (door_component_labels == door_id)
        dilated = ndimage.binary_dilation(door_pixels, iterations=3)
        touching_room_ids = set(room_labels[dilated & (room_labels > 0)].tolist())
        centroid = ndimage.center_of_mass(door_pixels)

        door_info[door_id] = {
            "centroid": centroid,
            "connects_rooms": touching_room_ids,
            "is_exit_candidate": False,
        }

    return door_info


def add_space_type_to_graph(G, space_type_map):
    """그래프 노드마다 room/corridor/stairs 중 무엇인지 속성을 부여."""
    h, w = space_type_map.shape
    for node_id, data in G.nodes(data=True):
        r = int(np.clip(round(data["row"]), 0, h - 1))
        c = int(np.clip(round(data["col"]), 0, w - 1))
        type_id = int(space_type_map[r, c])
        G.nodes[node_id]["space_type"] = SPACE_TYPE_NAMES[type_id]
    return G


# ------------------------------------------------------------
# 방(room) 전용 노드 생성 (중앙 지점 방식)
# ------------------------------------------------------------
def build_room_only_submask(space_type_map):
    """room(space_type==1) 픽셀만 FLOOR, 나머지는 전부 WALL인 서브마스크."""
    room_only = np.where(space_type_map == 1, tp.FLOOR, tp.WALL).astype(np.uint8)
    return room_only


def filter_small_rooms(room_labels_only, min_size=100):
    """
    점처럼 작은 노이즈성 방 조각(오예측 등)을 제거하고, 남은 방들의
    번호를 1부터 다시 연속되게 매김.
    """
    num_labels = int(room_labels_only.max())
    if num_labels == 0:
        return room_labels_only, 0, 0

    sizes = ndimage.sum(
        np.ones_like(room_labels_only), room_labels_only, index=range(1, num_labels + 1)
    )

    kept_ids = [i for i, size in enumerate(sizes, start=1) if size >= min_size]
    removed_count = num_labels - len(kept_ids)

    relabeled = np.zeros_like(room_labels_only)
    for new_id, old_id in enumerate(kept_ids, start=1):
        relabeled[room_labels_only == old_id] = new_id

    return relabeled, len(kept_ids), removed_count


def compute_room_center_nodes(room_labels_only, num_rooms, space_type_map):
    """
    방마다 노드를 하나씩만 생성. 기하학적 중심이 아니라 distance transform으로
    계산한 '벽에서 가장 먼 지점'을 씀 (오목한 방 모양이어도 안전).
    """
    room_nodes = {}
    for room_id in range(1, num_rooms + 1):
        room_mask = (room_labels_only == room_id)
        if not room_mask.any():
            continue

        dist = ndimage.distance_transform_edt(room_mask)
        r, c = np.unravel_index(np.argmax(dist), dist.shape)

        type_id = int(space_type_map[r, c])
        room_nodes[room_id] = {
            "row": float(r),
            "col": float(c),
            "space_type": SPACE_TYPE_NAMES.get(type_id, "room"),
            "area": int(room_mask.sum()),
        }
    return room_nodes


# ------------------------------------------------------------
# 복도(corridor)/계단(stairs)/문(door) 전용 스켈레톤 그래프 (기존 방식)
# ------------------------------------------------------------
def clean_and_skeletonize_preserving_stairs(corridor_submask, space_type_map):
    """
    tp.clean_and_skeletonize()와 거의 동일하지만, Morphology opening 단계에서
    계단(stairs)처럼 작거나 얇은 영역이 통째로 침식되어 사라지는 문제를 막기 위해
    opening 이후 계단 픽셀을 강제로 다시 살려서 skeletonize에 넘김.
    """
    walkable = (corridor_submask == tp.FLOOR) | (corridor_submask == tp.DOOR)

    cleaned = ndimage.binary_closing(walkable, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_opening(cleaned, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_fill_holes(cleaned)

    # opening 과정에서 계단 픽셀이 지워졌더라도 다시 복원 (계단은 얇아도 절대 사라지면 안 됨)
    stairs_mask = (space_type_map == 3)
    cleaned = cleaned | stairs_mask

    skeleton = tp.skeletonize(cleaned)
    return skeleton


def build_corridor_submask(mask, space_type_map):
    """corridor/stairs 픽셀 + door 픽셀만 살리고, room/wall은 전부 WALL 처리."""
    corridor_submask = np.full(mask.shape, tp.WALL, dtype=np.uint8)
    corridor_submask[(space_type_map == 2) | (space_type_map == 3)] = tp.FLOOR
    corridor_submask[mask == tp.DOOR] = tp.DOOR
    return corridor_submask


def compute_door_room_touches(mask, room_labels_only):
    """각 문이 어느 방(room_labels_only 기준)과 맞닿아 있는지 계산."""
    door_component_labels, num_doors = ndimage.label(mask == tp.DOOR)
    door_room_map = {}
    for door_id in range(1, num_doors + 1):
        door_pixels = (door_component_labels == door_id)
        dilated = ndimage.binary_dilation(door_pixels, iterations=3)
        touching_rooms = set(room_labels_only[dilated & (room_labels_only > 0)].tolist())
        centroid = ndimage.center_of_mass(door_pixels)
        door_room_map[door_id] = {"centroid": centroid, "touching_rooms": touching_rooms}
    return door_room_map


def connect_rooms_to_corridor_graph(G, corridor_G, room_nodes, door_room_map):
    """
    방 중앙 노드를, 그 방과 맞닿은 문에서 가장 가까운 복도 스켈레톤 노드에 연결.
    (방 중앙 -> 문 -> 가장 가까운 복도 노드) 순서의 거리 합을 엣지 가중치로 사용.
    """
    connected_count = 0
    for door_id, info in door_room_map.items():
        dr, dc = info["centroid"]

        nearest_node, best_dist = None, float("inf")
        for node, data in corridor_G.nodes(data=True):
            dist = ((data["row"] - dr) ** 2 + (data["col"] - dc) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                nearest_node = node

        if nearest_node is None:
            continue

        for room_id in info["touching_rooms"]:
            room_node = f"room_{room_id}"
            if room_node not in G:
                continue
            rr, rc = room_nodes[room_id]["row"], room_nodes[room_id]["col"]
            room_to_door = ((rr - dr) ** 2 + (rc - dc) ** 2) ** 0.5
            G.add_edge(room_node, nearest_node, weight=room_to_door + best_dist)
            connected_count += 1

    return connected_count


def visualize_real(mask, room_labels_only, skeleton, room_nodes, G, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(mask, cmap="tab10", vmin=0, vmax=2)
    axes[0, 0].set_title("1. Segmentation Mask (wall / floor / door)")
    axes[0, 0].axis("off")

    num_rooms = int(room_labels_only.max())
    perm = np.random.RandomState(0).permutation(num_rooms + 1)
    shuffled = perm[room_labels_only]
    shuffled[room_labels_only == 0] = 0
    axes[0, 1].imshow(shuffled, cmap="nipy_spectral")
    axes[0, 1].set_title(f"2. Room Labeling ({num_rooms} rooms, shuffled colors)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mask, cmap="gray", alpha=0.3)
    ys, xs = np.where(skeleton)
    axes[1, 0].scatter(xs, ys, s=1, c="red")
    for room_id, data in room_nodes.items():
        axes[1, 0].scatter(data["col"], data["row"], c="black", s=25, marker="x", zorder=3)
    axes[1, 0].set_title("3. Corridor skeleton (red) + Room centers (black x)")
    axes[1, 0].axis("off")

    type_colors = {
        "room": "#2ca02c",
        "corridor": "#9467bd",
        "stairs": "#17becf",
        "none": "#7f7f7f",
    }

    axes[1, 1].imshow(mask, cmap="gray", alpha=0.3)
    for u, v in G.edges:
        r1, c1 = G.nodes[u]["row"], G.nodes[u]["col"]
        r2, c2 = G.nodes[v]["row"], G.nodes[v]["col"]
        axes[1, 1].plot([c1, c2], [r1, r2], c="lightgray", linewidth=0.6, zorder=1)

    for node, data in G.nodes(data=True):
        if data.get("is_door"):
            color, size, z = "orange", 45, 3
        elif data.get("space_type") == "room":
            color, size, z = type_colors["room"], 80, 2
        else:
            color = type_colors.get(data.get("space_type", "none"), "#7f7f7f")
            size, z = 12, 2
        axes[1, 1].scatter(
            data["col"], data["row"], c=color, s=size, zorder=z,
            edgecolors="black" if data.get("space_type") == "room" else "none", linewidths=0.5,
        )

    legend_handles = [
        mpatches.Patch(color="orange", label="Door node"),
        mpatches.Patch(color=type_colors["room"], label="Room center node"),
        mpatches.Patch(color=type_colors["corridor"], label="Corridor node"),
        mpatches.Patch(color=type_colors["stairs"], label="Stairs node"),
    ]
    axes[1, 1].legend(handles=legend_handles, loc="upper right", fontsize=8)
    axes[1, 1].set_title("4. Final Graph (room=center node, corridor/door=skeleton nodes)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"결과 이미지 저장 완료: {os.path.abspath(save_path)}")


def analyze_floorplan(
    image_path, model_path="unet_finetuned.pth", image_size=512,
    min_door_size=25, max_door_area=200, door_erode_iterations=2,
    wall_dilate=1, door_match_radius=15, min_room_size=100,
    max_corridor_edge_length=40, door_merge_radius=15, stairs_merge_radius=20,
    device=None, model=None, mode="rule", rule_node_spacing=30,
):
    """
    도면 이미지 하나를 받아서 최종 그래프(G)까지 만들어서 반환.
    CLI(main())와 FastAPI 서비스 양쪽에서 재사용하기 위해 분리한 핵심 로직.

    model을 미리 로딩해서 넘기면(FastAPI에서 매 요청마다 모델을 다시
    불러오지 않도록) 그걸 재사용하고, 안 넘기면 model_path로 새로 로딩함.

    mode="rule"(기본): rule_based_floorplan.py의 규칙 기반 분석 (벽선을 직접 읽음, 모델 미사용).
        분석 중 예외가 나거나 노드를 하나도 찾지 못하면 아래 U-Net 방식으로 대체해서
        항상 기존과 같은 형식의 결과를 돌려줌.
    mode="unet": 기존 U-Net 세그멘테이션 기반 분석.
    rule_node_spacing: 규칙 기반 방식의 복도 노드 간격 (image_size=512 기준 픽셀).

    어느 방식이 실제로 쓰였는지는 반환값의 stats["mode"]에 기록됨
    ("rule" / "unet" / "unet_fallback").

    반환값: dict(G, mask, room_labels_only, skeleton, room_nodes, stats)
    """
    analysis_mode = "unet"
    if mode == "rule":
        try:
            from rule_based_floorplan import analyze_floorplan_rule_based
            # rule_node_spacing은 image_size=512 기준 값. 요청의 image_size가 달라도
            # 도면 대비 노드 간격(=노드 개수)이 같게 나오도록 비율을 맞춤.
            rule_result = analyze_floorplan_rule_based(
                image_path, image_size=image_size,
                corridor_node_spacing=rule_node_spacing * image_size / 512,
            )
            if rule_result["G"].number_of_nodes() > 0:
                return rule_result
            print("경고: 규칙 기반 분석이 노드를 찾지 못해 U-Net 방식으로 대체합니다.")
        except Exception as e:
            print(f"경고: 규칙 기반 분석 실패 ({type(e).__name__}: {e}) - U-Net 방식으로 대체합니다.")
        analysis_mode = "unet_fallback"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = UNet(in_channels=3, num_classes=6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    pred_6class = predict_mask_6class(model, image_path, device, image_size=image_size)
    space_type_map = build_space_type_map(pred_6class)
    mask = remap_to_pipeline_classes(pred_6class)

    mask, total_doors, removed_doors = clean_door_noise(mask, min_size=min_door_size)
    mask, num_shrunk = shrink_oversized_doors(
        mask, max_area=max_door_area, erosion_iterations=door_erode_iterations
    )
    mask = reinforce_walls(mask, iterations=wall_dilate)

    # ---- 방 전용 라벨링 + 중앙 노드 ----
    room_submask = build_room_only_submask(space_type_map)
    room_labels_only, num_rooms_only = tp.label_rooms(room_submask)
    room_labels_only, num_rooms_only, removed_rooms = filter_small_rooms(
        room_labels_only, min_size=min_room_size
    )
    room_nodes = compute_room_center_nodes(room_labels_only, num_rooms_only, space_type_map)

    # ---- 복도/계단/문 전용 스켈레톤 그래프 ----
    corridor_submask = build_corridor_submask(mask, space_type_map)
    corridor_region_labels, num_corridor_regions = tp.label_rooms(corridor_submask)
    door_info = compute_door_connections(mask, corridor_region_labels)

    skeleton = clean_and_skeletonize_preserving_stairs(corridor_submask, space_type_map)
    raw_graph = tp.skeleton_to_graph(skeleton)
    corridor_G = tp.simplify_graph(raw_graph)
    corridor_G = tp.annotate_graph(corridor_G, corridor_region_labels, door_info,
                                    door_match_radius=door_match_radius)
    corridor_G = add_space_type_to_graph(corridor_G, space_type_map)
    corridor_G = densify_long_edges(corridor_G, max_edge_length=max_corridor_edge_length)
    corridor_G = nx.relabel_nodes(corridor_G, {n: f"corridor_{n}" for n in corridor_G.nodes})

    # ---- 최종 그래프 합치기 ----
    G = nx.Graph()
    for room_id, data in room_nodes.items():
        G.add_node(
            f"room_{room_id}",
            row=data["row"], col=data["col"],
            room_id=room_id, space_type=data["space_type"], is_door=False,
        )
    G.add_nodes_from(corridor_G.nodes(data=True))
    G.add_edges_from(corridor_G.edges(data=True))

    door_room_map = compute_door_room_touches(mask, room_labels_only)
    connect_rooms_to_corridor_graph(G, corridor_G, room_nodes, door_room_map)

    G, merged_door_count = merge_nearby_door_nodes(G, merge_radius=door_merge_radius)
    G, merged_stairs_count = merge_nearby_stairs_nodes(G, merge_radius=stairs_merge_radius)

    return {
        "G": G,
        "mask": mask,
        "room_labels_only": room_labels_only,
        "skeleton": skeleton,
        "room_nodes": room_nodes,
        "stats": {
            "mode": analysis_mode,
            "total_doors": total_doors,
            "removed_doors": removed_doors,
            "shrunk_doors": num_shrunk,
            "num_rooms": num_rooms_only,
            "removed_small_rooms": removed_rooms,
            "merged_door_nodes": merged_door_count,
            "merged_stairs_nodes": merged_stairs_count,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rule", choices=["rule", "unet"],
                        help="rule: 벽선을 직접 읽는 규칙 기반 (기본) / unet: 학습된 U-Net 사용")
    parser.add_argument("--model_path", type=str, default="unet_finetuned.pth")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="real_floorplan_pipeline_result.png")
    parser.add_argument("--min_door_size", type=int, default=25,
                        help="이보다 작은 문 조각은 노이즈로 간주하고 FLOOR로 되돌림")
    parser.add_argument("--max_door_area", type=int, default=200,
                        help="이보다 큰 문 조각은 과잉예측으로 간주하고 침식시켜 줄임")
    parser.add_argument("--door_erode_iterations", type=int, default=2,
                        help="과대 문 조각을 줄일 때 침식 반복 횟수")
    parser.add_argument("--wall_dilate", type=int, default=1,
                        help="예측된 벽을 이 픽셀 수만큼 두껍게 보강 (0이면 비활성화)")
    parser.add_argument("--door_match_radius", type=int, default=15,
                        help="복도 스켈레톤 노드를 문으로 인식하는 반경(픽셀)")
    parser.add_argument("--min_room_size", type=int, default=100,
                        help="이 픽셀 수보다 작은 방 조각은 노이즈로 간주하고 제외")
    parser.add_argument("--max_corridor_edge_length", type=int, default=40,
                        help="복도 엣지가 이보다 길면 중간에 노드를 추가로 끼워 넣음")
    parser.add_argument("--door_merge_radius", type=int, default=15,
                        help="이 반경(픽셀) 안에 있는 문 노드들은 하나로 병합")
    parser.add_argument("--stairs_merge_radius", type=int, default=20,
                        help="이 반경(픽셀) 안에 있는 계단 노드들은 하나로 병합")
    parser.add_argument("--export_db", action="store_true",
                        help="완성된 그래프를 PostgreSQL에 저장 (db_export.py, schema.sql 필요)")
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_port", type=int, default=5432)
    parser.add_argument("--db_name", type=str, default="saferoute")
    parser.add_argument("--db_user", type=str, default="postgres")
    parser.add_argument("--db_password", type=str, default="")
    parser.add_argument("--floorplan_name", type=str, default=None,
                        help="DB에 저장할 도면 이름 (지정 안 하면 이미지 파일명 사용)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    result = analyze_floorplan(
        image_path=args.image_path, model_path=args.model_path, image_size=args.image_size,
        min_door_size=args.min_door_size, max_door_area=args.max_door_area,
        door_erode_iterations=args.door_erode_iterations, wall_dilate=args.wall_dilate,
        door_match_radius=args.door_match_radius, min_room_size=args.min_room_size,
        max_corridor_edge_length=args.max_corridor_edge_length,
        door_merge_radius=args.door_merge_radius, stairs_merge_radius=args.stairs_merge_radius,
        device=device, mode=args.mode,
    )
    print(f"   -> 분석 방식: {result['stats'].get('mode')}, 통계: {result['stats']}")
    G = result["G"]
    mask = result["mask"]
    room_labels_only = result["room_labels_only"]
    skeleton = result["skeleton"]
    room_nodes = result["room_nodes"]

    print(f"   -> 전체 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
    type_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("space_type", "none")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"   -> 노드 타입 분포: {type_counts}")

    isolated = list(nx.isolates(G))
    if isolated:
        print(f"   -> 주의: 어떤 문과도 연결되지 않은 고립 노드 {len(isolated)}개 있음: {isolated}")

    print("결과 시각화...")
    visualize_real(mask, room_labels_only, skeleton, room_nodes, G, save_path=args.save_path)

    if result["stats"].get("mode") == "rule":
        # 원본 도면 위에 방/문/복도 노드를 그린 검토용 그림 (원본 비율 그대로)
        from rule_based_floorplan import save_overlay
        overlay_path = os.path.splitext(args.save_path)[0] + "_overlay.png"
        save_overlay(args.image_path, result, overlay_path)
        print(f"검토용 오버레이 저장 완료: {os.path.abspath(overlay_path)}")

    if args.export_db:
        if not DB_EXPORT_AVAILABLE:
            print("경고: db_export.py를 찾을 수 없거나 psycopg2가 설치되지 않아 DB 저장을 건너뜁니다.")
            print("       pip install psycopg2-binary 후 다시 시도해주세요.")
        else:
            print("PostgreSQL에 그래프 저장 중...")
            floorplan_name = args.floorplan_name or os.path.basename(args.image_path)
            db_config = {
                "host": args.db_host, "port": args.db_port, "dbname": args.db_name,
                "user": args.db_user, "password": args.db_password,
            }
            export_graph_to_postgres(
                G, floorplan_name=floorplan_name, image_path=args.image_path,
                image_size=args.image_size, db_config=db_config,
            )


if __name__ == "__main__":
    main()