"""
도면 이미지 -> 대피 경로 그래프 변환.
U-Net(6클래스 세그멘테이션)으로 픽셀을 분류하고, 후처리로 방/복도/문/계단을 나눠 노드와 엣지를 만든다.
규칙 기반 분석(rule_based_floorplan)은 mode="rule"로 선택.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import UNet
import test_pipeline as tp

try:
    from db_export import export_graph_to_postgres
    DB_EXPORT_AVAILABLE = True
except ImportError:
    DB_EXPORT_AVAILABLE = False


# 학습 때 쓴 클래스 순서
FT_WALL, FT_ROOM, FT_CORRIDOR, FT_DOOR, FT_WINDOW, FT_STAIRS = range(6)
SPACE_TYPE_NAMES = {0: "none", 1: "room", 2: "corridor", 3: "stairs"}

# 6클래스 -> test_pipeline의 WALL/FLOOR/DOOR
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

    return pred_mask


def remap_to_pipeline_classes(pred_mask_6class):
    remapped = np.zeros_like(pred_mask_6class, dtype=np.uint8)
    for src_cls, dst_cls in CLASS_REMAP.items():
        remapped[pred_mask_6class == src_cls] = dst_cls
    return remapped


def reinforce_walls(mask, iterations=1):
    """벽을 FLOOR 쪽으로 팽창시켜 끊긴 곳을 잇는다. DOOR는 건드리지 않음."""
    if iterations <= 0:
        return mask
    wall_mask = (mask == tp.WALL)
    dilated_wall = ndimage.binary_dilation(wall_mask, iterations=iterations)
    overwrite = dilated_wall & (mask == tp.FLOOR)
    reinforced = mask.copy()
    reinforced[overwrite] = tp.WALL
    return reinforced


def build_space_type_map(pred_mask_6class):
    """room=1, corridor=2, stairs=3, 나머지 0."""
    space_type = np.zeros_like(pred_mask_6class, dtype=np.uint8)
    space_type[pred_mask_6class == FT_ROOM] = 1
    space_type[pred_mask_6class == FT_CORRIDOR] = 2
    space_type[pred_mask_6class == FT_STAIRS] = 3
    return space_type


def seal_wall_gaps_6class(pred, iterations):
    """벽을 방/복도/계단 쪽으로 팽창시켜 끊긴 곳을 메운다. 문은 유지."""
    if iterations <= 0:
        return pred
    dilated = ndimage.binary_dilation(pred == FT_WALL, iterations=iterations)
    out = pred.copy()
    out[dilated & np.isin(pred, [FT_ROOM, FT_CORRIDOR, FT_STAIRS])] = FT_WALL
    return out


def fill_room_blotches_6class(pred, max_hole_px):
    """방 안의 작은 비(非)방 조각을 방으로 메운다."""
    if max_hole_px <= 0:
        return pred
    room = pred == FT_ROOM
    labels, n = ndimage.label(~room)
    if n == 0:
        return pred
    sizes = np.bincount(labels.ravel())[1:]
    small = np.isin(labels, np.where(sizes < max_hole_px)[0] + 1)
    out = pred.copy()
    out[small & (pred != FT_DOOR)] = FT_ROOM
    return out


def building_footprint_mask(image_path, out_shape, door_gap_frac=0.07, min_area_frac=0.2):
    """
    도면의 굵은 선을 문 틈 폭만큼 닫고 안쪽을 채워 건물 내부 마스크를 만든다.
    면적이 너무 작으면 실패로 보고 None.
    """
    gray = np.array(Image.open(image_path).convert("L")).astype(np.int16)
    h, w = gray.shape
    dark = gray < 128
    k = max(2, round(w / 400))
    thick = ndimage.binary_opening(dark, structure=np.ones((k, k)))
    gap = max(5, round(door_gap_frac * w))
    padded = np.pad(thick, gap)
    sealed = ndimage.binary_closing(padded, structure=np.ones((3, 3)), iterations=gap // 2)[gap:-gap, gap:-gap]
    footprint = ndimage.binary_fill_holes(sealed)
    if footprint.mean() < min_area_frac:
        return None
    fp = Image.fromarray(footprint.astype(np.uint8) * 255).resize((out_shape[1], out_shape[0]), Image.NEAREST)
    return np.array(fp) > 0


def absorb_corridor_like_rooms(room_labels, space_type_map, min_corridor_contact=0.35, max_area=None):
    """테두리의 상당 부분이 복도와 바로 닿는 작은 방 조각을 복도로 돌린다."""
    absorbed = 0
    for room_id in range(1, int(room_labels.max()) + 1):
        comp = room_labels == room_id
        if not comp.any():
            continue
        if max_area is not None and comp.sum() > max_area:
            continue
        ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
        if ring.sum() == 0:
            continue
        if float((space_type_map[ring] == 2).mean()) >= min_corridor_contact:
            space_type_map[comp] = 2
            room_labels[comp] = 0
            absorbed += 1
    return room_labels, space_type_map, absorbed


def remove_unsupported_walls(pred, image_path, footprint, ink_thresh=170, ink_dilate=3, min_thickness=2.0):
    """
    원본에 선이 없는데 벽으로 예측된 두꺼운 덩어리를 지우고 주변 클래스로 채운다.
    실제 벽 가장자리의 얇은 오차는 min_thickness 미만이라 남는다.
    """
    h, w = pred.shape
    gray = np.array(Image.open(image_path).convert("L").resize((w, h), Image.BOX))
    ink = ndimage.binary_dilation(gray < ink_thresh, iterations=ink_dilate)
    fake = (pred == FT_WALL) & ~ink
    if footprint is not None:
        fake &= footprint
    if not fake.any():
        return pred, 0
    if min_thickness > 0:
        dt = ndimage.distance_transform_edt(fake)
        labels, n = ndimage.label(fake)
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = np.asarray(ndimage.maximum(dt, labels, index=range(1, n + 1))) >= min_thickness
        fake = keep[labels]
    if not fake.any():
        return pred, 0
    _, idx = ndimage.distance_transform_edt(pred == FT_WALL, return_indices=True)
    out = pred.copy()
    out[fake] = pred[idx[0][fake], idx[1][fake]]
    return out, int(fake.sum())


def clean_door_noise(mask, min_size=25):
    """너무 작은 문 조각은 FLOOR로 되돌린다."""
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
    """너무 큰 문 덩어리는 침식해서 줄인다."""
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
    """max_edge_length보다 긴 엣지 사이에 중간 노드를 끼워 넣는다."""
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
    """node_filter에 걸리는 노드 중 merge_radius 안에 몰린 것들을 하나로 합친다."""
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
    return merge_nearby_nodes(G, lambda d: d.get("is_door"), merge_radius)


def merge_nearby_stairs_nodes(G, merge_radius=20):
    return merge_nearby_nodes(G, lambda d: d.get("space_type") == "stairs", merge_radius)


def compute_door_connections(mask, room_labels):
    """문 조각마다 맞닿은 영역 id 집합과 중심을 구한다."""
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
    """스켈레톤 노드에 space_type을 붙인다. 방 픽셀 위라도 복도로 둔다(방 노드는 따로 만듦)."""
    h, w = space_type_map.shape
    for node_id, data in G.nodes(data=True):
        r = int(np.clip(round(data["row"]), 0, h - 1))
        c = int(np.clip(round(data["col"]), 0, w - 1))
        type_id = int(space_type_map[r, c])
        space_type = SPACE_TYPE_NAMES[type_id]
        if space_type == "room":
            space_type = "corridor"
        G.nodes[node_id]["space_type"] = space_type
    return G


def build_room_only_submask(space_type_map):
    room_only = np.where(space_type_map == 1, tp.FLOOR, tp.WALL).astype(np.uint8)
    return room_only


def filter_small_rooms(room_labels_only, min_size=100):
    """min_size 미만 조각을 버리고 1부터 다시 번호를 매긴다."""
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
    """방마다 벽에서 가장 먼 지점에 노드 하나."""
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


def clean_and_skeletonize_preserving_stairs(corridor_submask, space_type_map):
    """모폴로지 정리 후 스켈레톤화. opening에서 지워진 계단 픽셀은 복원한다."""
    walkable = (corridor_submask == tp.FLOOR) | (corridor_submask == tp.DOOR)

    cleaned = ndimage.binary_closing(walkable, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_opening(cleaned, structure=np.ones((3, 3)))
    cleaned = ndimage.binary_fill_holes(cleaned)

    stairs_mask = (space_type_map == 3)
    cleaned = cleaned | stairs_mask

    skeleton = tp.skeletonize(cleaned)
    return skeleton


def build_corridor_submask(mask, space_type_map):
    """복도/계단/문만 남긴 3클래스 마스크."""
    corridor_submask = np.full(mask.shape, tp.WALL, dtype=np.uint8)
    corridor_submask[(space_type_map == 2) | (space_type_map == 3)] = tp.FLOOR
    corridor_submask[mask == tp.DOOR] = tp.DOOR
    return corridor_submask


def compute_door_room_touches(mask, room_labels_only):
    """문 조각마다 맞닿은 방 id 집합과 중심."""
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
    """방 노드를 문에서 가장 가까운 복도 노드에 연결한다. 가중치는 방->문->복도 거리 합."""
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


def label_rooms_watershed(space_type_map, h=2.0):
    """방 픽셀을 방 단위로 라벨링. h<=0이면 연결 성분, 아니면 h-maxima 씨앗 watershed."""
    room = (space_type_map == 1)
    room = ndimage.binary_closing(room, structure=np.ones((3, 3)))
    if not room.any():
        return np.zeros(room.shape, dtype=np.int32)
    if h <= 0:
        labels, _ = ndimage.label(room)
        return labels.astype(np.int32)

    dist = ndimage.distance_transform_edt(room)
    maxima = h_maxima(dist, h)
    markers, _ = ndimage.label(maxima)

    # 씨앗이 없는 성분은 그 성분의 최댓점을 씨앗으로
    components, num_components = ndimage.label(room)
    for comp_id, slc in enumerate(ndimage.find_objects(components), start=1):
        comp_mask = (components[slc] == comp_id)
        if markers[slc][comp_mask].any():
            continue
        local_dist = np.where(comp_mask, dist[slc], -1)
        r, c = np.unravel_index(np.argmax(local_dist), local_dist.shape)
        markers[slc][r, c] = markers.max() + 1

    return watershed(-dist, markers, mask=room).astype(np.int32)


def prune_spurs(G, min_len=10):
    """분기점에 붙은 min_len 미만의 잔가지를 제거."""
    removed = 0
    changed = True
    while changed:
        changed = False
        for leaf in [n for n, d in G.degree() if d == 1]:
            if leaf not in G or G.degree(leaf) != 1:
                continue
            path, length, prev, cur = [leaf], 0.0, None, leaf
            while True:
                nbrs = [x for x in G.neighbors(cur) if x != prev]
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                length += G[cur][nxt].get("weight", 1.0)
                prev, cur = cur, nxt
                path.append(cur)
                if G.degree(cur) != 2:
                    break
            if G.degree(cur) >= 3 and length < min_len:
                G.remove_nodes_from(path[:-1])
                removed += 1
                changed = True
    return G, removed


def mark_door_nodes(corridor_G, mask, radius=15):
    """문 조각마다 가장 가까운 복도 노드 하나에만 is_door를 켠다."""
    for _, data in corridor_G.nodes(data=True):
        data["is_door"] = False

    door_labels, num_doors = ndimage.label(mask == tp.DOOR)
    if num_doors == 0 or corridor_G.number_of_nodes() == 0:
        return corridor_G

    centroids = ndimage.center_of_mass(mask == tp.DOOR, door_labels, range(1, num_doors + 1))
    nodes = list(corridor_G.nodes(data=True))
    coords = np.array([[d["row"], d["col"]] for _, d in nodes])
    for dr, dc in centroids:
        dist = np.hypot(coords[:, 0] - dr, coords[:, 1] - dc)
        i = int(np.argmin(dist))
        if dist[i] <= radius:
            nodes[i][1]["is_door"] = True
    return corridor_G


def filter_small_stairs(space_type_map, min_area=300):
    """min_area 미만의 계단 조각은 복도로 돌린다."""
    stairs = (space_type_map == 3)
    labels, num = ndimage.label(stairs)
    out = space_type_map.copy()
    removed = 0
    if num == 0:
        return out, 0
    sizes = ndimage.sum(stairs, labels, index=range(1, num + 1))
    for stair_id, size in enumerate(sizes, start=1):
        if size < min_area:
            out[labels == stair_id] = 2
            removed += 1
    return out, removed


def drop_internal_doors(mask, space_type_map, room_labels_only):
    """복도와 닿지 않고 방 하나에만 붙은 문 조각은 문이 아니라고 보고 바닥으로 돌린다."""
    door_labels, n = ndimage.label(mask == tp.DOOR)
    removed = 0
    pad = 4
    for door_id, slc in enumerate(ndimage.find_objects(door_labels), start=1):
        if slc is None:
            continue
        ys = slice(max(0, slc[0].start - pad), min(mask.shape[0], slc[0].stop + pad))
        xs = slice(max(0, slc[1].start - pad), min(mask.shape[1], slc[1].stop + pad))
        comp = door_labels[ys, xs] == door_id
        ring = ndimage.binary_dilation(comp, iterations=3) & ~comp
        rooms = set(room_labels_only[ys, xs][ring].tolist()) - {0}
        touches_corridor = bool((space_type_map[ys, xs][ring] >= 2).any())
        if touches_corridor or len(rooms) >= 2:
            continue
        mask[ys, xs][comp] = tp.FLOOR
        space_type_map[ys, xs][comp] = 1 if rooms else 0
        removed += 1
    return mask, space_type_map, removed


def connect_isolated_rooms(G, max_dist=80):
    """연결이 없는 방 노드를 max_dist 안의 가장 가까운 복도 노드에 잇고, 고립된 복도 노드는 지운다."""
    corridor_nodes = [n for n in G.nodes if str(n).startswith("corridor_")]
    connected_rooms = 0
    for node in list(G.nodes):
        if not str(node).startswith("room_") or G.degree(node) > 0:
            continue
        rr, rc = G.nodes[node]["row"], G.nodes[node]["col"]
        best, best_dist = None, float("inf")
        for c in corridor_nodes:
            if G.nodes[c].get("space_type") == "none":
                continue
            dist = ((G.nodes[c]["row"] - rr) ** 2 + (G.nodes[c]["col"] - rc) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist = c, dist
        if best is not None and best_dist <= max_dist:
            G.add_edge(node, best, weight=best_dist)
            connected_rooms += 1

    orphan_corridors = [n for n in corridor_nodes if G.degree(n) == 0]
    G.remove_nodes_from(orphan_corridors)
    return G, connected_rooms, len(orphan_corridors)


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
    image_path, model_path="unet_finetuned_v8.pth", image_size=512,
    min_door_size=25, max_door_area=200, door_erode_iterations=2,
    wall_dilate=2, door_match_radius=15, min_room_size=800,
    max_corridor_edge_length=30, door_merge_radius=15, room_split_h=0.0,
    spur_min_len=10, min_stairs_area=300, isolated_room_max_dist=80,
    device=None, model=None, mode="unet", rule_node_spacing=30, stairs_merge_radius=20,
    unet_infer_size=768, room_hole_fill=300, drop_exterior=True, corridor_absorb=0.35,
    remove_fake_walls=True, corridor_absorb_max_area=1500,
    room_enclosure_check=True, room_gap_frac=0.04, corridor_merge_radius=20,
    exterior_stairs_max_dist=90,
):
    """
    도면 한 장을 분석해 dict(G, mask, room_labels_only, skeleton, room_nodes, stats)를 돌려준다.
    mode="unet"(기본)은 U-Net 세그멘테이션 + 후처리, mode="rule"은 규칙 기반이며 실패 시 U-Net으로 대체.
    실제 쓰인 방식은 stats["mode"]에 남는다.
    """
    analysis_mode = "unet"
    if mode == "rule":
        try:
            from rule_based_floorplan import analyze_floorplan_rule_based
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

    # 픽셀 기준값들은 512 프레임 기준. 처리는 512로 하고 마지막에 좌표만 요청 크기로 환산
    requested_size = image_size
    image_size = 512

    # 얇은 벽이 사라지지 않게 더 큰 해상도로 예측한 뒤 축소
    infer_size = max(image_size, unet_infer_size or image_size)
    infer_size -= infer_size % 8
    pred_hr = predict_mask_6class(model, image_path, device, image_size=infer_size)
    hr_scale = (infer_size * infer_size) / (image_size * image_size)
    pred_hr = seal_wall_gaps_6class(pred_hr, iterations=wall_dilate)
    pred_hr = fill_room_blotches_6class(pred_hr, max_hole_px=int(room_hole_fill * hr_scale))
    dropped_exterior = False
    footprint = None
    exterior_stairs = []  # (중심 row, 중심 col, 계단 영역까지의 거리 맵) image_size 프레임
    if drop_exterior:
        footprint = building_footprint_mask(image_path, pred_hr.shape)
        if footprint is not None:
            ext = (pred_hr == FT_STAIRS) & ~footprint
            grouped, n_groups = ndimage.label(ndimage.binary_dilation(ext, iterations=25))
            for gid in range(1, n_groups + 1):
                comp = ext & (grouped == gid)
                if comp.sum() >= min_stairs_area * hr_scale:
                    r, c = ndimage.center_of_mass(comp)
                    comp_small = np.array(Image.fromarray(comp.astype(np.uint8) * 255).resize(
                        (image_size, image_size), Image.NEAREST)) > 0
                    dist_map = ndimage.distance_transform_edt(~comp_small)
                    exterior_stairs.append((r * image_size / infer_size, c * image_size / infer_size, dist_map))
            pred_hr = pred_hr.copy()
            pred_hr[~footprint & (pred_hr != FT_WALL)] = FT_WALL
            dropped_exterior = True
    removed_fake_wall_px = 0
    if remove_fake_walls:
        pred_hr, removed_fake_wall_px = remove_unsupported_walls(pred_hr, image_path, footprint)
    if infer_size != image_size:
        pred_6class = np.array(Image.fromarray(pred_hr.astype(np.uint8)).resize(
            (image_size, image_size), Image.NEAREST))
    else:
        pred_6class = pred_hr
    space_type_map = build_space_type_map(pred_6class)
    space_type_map, removed_stairs = filter_small_stairs(space_type_map, min_area=min_stairs_area)
    mask = remap_to_pipeline_classes(pred_6class)

    mask, total_doors, removed_doors = clean_door_noise(mask, min_size=min_door_size)
    mask, num_shrunk = shrink_oversized_doors(
        mask, max_area=max_door_area, erosion_iterations=door_erode_iterations
    )

    # 방 라벨링
    room_labels_only = label_rooms_watershed(space_type_map, h=room_split_h)
    room_labels_only, num_rooms_only, removed_rooms = filter_small_rooms(
        room_labels_only, min_size=min_room_size
    )
    # 방은 문 폭 이하의 틈만 가진 닫힌 공간이어야 한다. 그 밖의 조각은 방에서 뺀다
    dropped_unenclosed = 0
    if room_enclosure_check:
        tight = building_footprint_mask(image_path, room_labels_only.shape, door_gap_frac=room_gap_frac)
        if tight is not None:
            for room_id in range(1, int(room_labels_only.max()) + 1):
                comp = room_labels_only == room_id
                if not comp.any() or tight[comp].mean() >= 0.5:
                    continue
                ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
                touches_corridor = ring.any() and (space_type_map[ring] == 2).mean() > 0.1
                space_type_map[comp] = 2 if touches_corridor else 0
                if not touches_corridor:
                    mask[comp] = tp.WALL
                room_labels_only[comp] = 0
                dropped_unenclosed += 1
            room_labels_only, num_rooms_only, _ = filter_small_rooms(room_labels_only, min_size=1)

    absorbed_rooms = 0
    if corridor_absorb and corridor_absorb > 0:
        room_labels_only, space_type_map, absorbed_rooms = absorb_corridor_like_rooms(
            room_labels_only, space_type_map, min_corridor_contact=corridor_absorb,
            max_area=corridor_absorb_max_area,
        )
        room_labels_only, num_rooms_only, _ = filter_small_rooms(room_labels_only, min_size=1)
    room_nodes = compute_room_center_nodes(room_labels_only, num_rooms_only, space_type_map)
    mask, space_type_map, removed_internal_doors = drop_internal_doors(mask, space_type_map, room_labels_only)

    # 복도/계단/문 스켈레톤 그래프
    corridor_submask = build_corridor_submask(mask, space_type_map)
    corridor_region_labels, num_corridor_regions = tp.label_rooms(corridor_submask)
    door_info = compute_door_connections(mask, corridor_region_labels)

    skeleton = clean_and_skeletonize_preserving_stairs(corridor_submask, space_type_map)
    raw_graph = tp.skeleton_to_graph(skeleton)
    corridor_G = tp.simplify_graph(raw_graph)
    corridor_G, _ = prune_spurs(corridor_G, min_len=spur_min_len)
    corridor_G = tp.simplify_graph(corridor_G)
    corridor_G = tp.annotate_graph(corridor_G, corridor_region_labels, door_info,
                                    door_match_radius=door_match_radius)
    corridor_G = mark_door_nodes(corridor_G, mask, radius=door_match_radius)
    corridor_G = add_space_type_to_graph(corridor_G, space_type_map)
    corridor_G = nx.relabel_nodes(corridor_G, {n: f"corridor_{n}" for n in corridor_G.nodes})

    # 방 안에 생긴 스켈레톤 조각 제거
    h_map, w_map = room_labels_only.shape
    inside_room = [
        n for n, d in corridor_G.nodes(data=True)
        if room_labels_only[
            int(np.clip(round(d["row"]), 0, h_map - 1)), int(np.clip(round(d["col"]), 0, w_map - 1))] > 0
    ]
    corridor_G.remove_nodes_from(inside_room)

    # 가까운 복도 노드는 합치고, 긴 구간만 다시 나눈다
    corridor_G, merged_corridor_count = merge_nearby_nodes(
        corridor_G, lambda d: not d.get("is_door") and d.get("space_type") != "stairs", corridor_merge_radius)
    for _, d in corridor_G.nodes(data=True):
        if d.get("space_type") == "none":
            d["space_type"] = "corridor"
    corridor_G = densify_long_edges(corridor_G, max_edge_length=max_corridor_edge_length)
    main_corridor = max(nx.connected_components(corridor_G), key=len) if corridor_G.number_of_nodes() else set()

    # 최종 그래프
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
    connect_rooms_to_corridor_graph(G, corridor_G.subgraph(main_corridor), room_nodes, door_room_map)
    # 방이 하나도 붙지 않은 복도 조각은 버린다
    removed_fragments = 0
    for comp in list(nx.connected_components(G)):
        if not any(str(n).startswith("room_") for n in comp):
            G.remove_nodes_from(comp)
            removed_fragments += 1
    G, rooms_fallback_connected, orphan_removed = connect_isolated_rooms(
        G, max_dist=isolated_room_max_dist
    )

    G, merged_count = merge_nearby_door_nodes(G, merge_radius=door_merge_radius)
    G, merged_stairs_count = merge_nearby_stairs_nodes(G, merge_radius=stairs_merge_radius)

    # 건물 밖 계단은 노드 하나로 만들어 가장 가까운 문/복도 노드에 잇는다
    exterior_stairs_added = 0
    for i, (sr, sc, dist_map) in enumerate(exterior_stairs, start=1):
        best, best_dist = None, float("inf")
        for n, d in G.nodes(data=True):
            if str(n).startswith("room_") or d.get("space_type") == "room":
                continue
            rr = int(np.clip(round(d["row"]), 0, image_size - 1))
            cc = int(np.clip(round(d["col"]), 0, image_size - 1))
            dist = float(dist_map[rr, cc])  # 노드에서 계단 영역 가장자리까지
            if dist < best_dist:
                best, best_dist = n, dist
        if best is not None and best_dist <= exterior_stairs_max_dist:
            node_id = f"stairs_ext_{i}"
            G.add_node(node_id, row=float(sr), col=float(sc), room_id=None, space_type="stairs", is_door=False)
            G.add_edge(node_id, best, weight=float(((G.nodes[best]["row"] - sr) ** 2 + (G.nodes[best]["col"] - sc) ** 2) ** 0.5))
            exterior_stairs_added += 1

    if requested_size != image_size:
        f = requested_size / image_size
        for _, data in G.nodes(data=True):
            data["row"] *= f
            data["col"] *= f
        for _, _, data in G.edges(data=True):
            if "weight" in data:
                data["weight"] *= f
        for data in room_nodes.values():
            data["row"] *= f
            data["col"] *= f
            data["area"] = int(data["area"] * f * f)

        def _resize(arr):
            return np.array(Image.fromarray(arr).resize((requested_size, requested_size), Image.NEAREST))

        mask = _resize(mask.astype(np.uint8))
        room_labels_only = _resize(room_labels_only.astype(np.int32))
        skeleton = _resize(skeleton.astype(np.uint8)) > 0

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
            "merged_door_nodes": merged_count,
            "merged_stairs_nodes": merged_stairs_count,
            "merged_corridor_nodes": merged_corridor_count,
            "removed_room_fragments": len(inside_room),
            "removed_internal_doors": removed_internal_doors,
            "exterior_stairs_nodes": exterior_stairs_added,
            "removed_corridor_fragments": removed_fragments,
            "removed_small_stairs": removed_stairs,
            "absorbed_corridor_rooms": absorbed_rooms,
            "removed_fake_wall_px": removed_fake_wall_px,
            "dropped_unenclosed_rooms": dropped_unenclosed,
            "dropped_exterior": dropped_exterior,
            "unet_infer_size": infer_size,
            "fallback_connected_rooms": rooms_fallback_connected,
            "removed_orphan_corridors": orphan_removed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="unet", choices=["unet", "rule"],
                        help="unet: U-Net 세그멘테이션 (기본) / rule: 규칙 기반")
    parser.add_argument("--unet_infer_size", type=int, default=768, help="U-Net 예측 해상도")
    parser.add_argument("--model_path", type=str, default="unet_finetuned_v8.pth")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--save_path", type=str, default="real_floorplan_pipeline_result.png")
    parser.add_argument("--min_door_size", type=int, default=25, help="이보다 작은 문 조각은 제거")
    parser.add_argument("--max_door_area", type=int, default=200, help="이보다 큰 문 조각은 침식")
    parser.add_argument("--door_erode_iterations", type=int, default=2)
    parser.add_argument("--room_split_h", type=float, default=0.0, help="0이면 연결 성분, >0이면 watershed h")
    parser.add_argument("--spur_min_len", type=int, default=10, help="이보다 짧은 복도 잔가지는 제거")
    parser.add_argument("--wall_dilate", type=int, default=2, help="벽 팽창 픽셀 수")
    parser.add_argument("--door_match_radius", type=int, default=15)
    parser.add_argument("--min_room_size", type=int, default=800, help="이보다 작은 방 조각은 제외")
    parser.add_argument("--max_corridor_edge_length", type=int, default=30, help="복도 노드 간격")
    parser.add_argument("--door_merge_radius", type=int, default=15)
    parser.add_argument("--stairs_merge_radius", type=int, default=20)
    parser.add_argument("--corridor_merge_radius", type=int, default=20, help="이 반경 안의 복도 노드는 하나로 합침")
    parser.add_argument("--export_db", action="store_true", help="PostgreSQL 저장 (db_export.py 필요)")
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_port", type=int, default=5432)
    parser.add_argument("--db_name", type=str, default="saferoute")
    parser.add_argument("--db_user", type=str, default="postgres")
    parser.add_argument("--db_password", type=str, default="")
    parser.add_argument("--floorplan_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    result = analyze_floorplan(
        image_path=args.image_path, model_path=args.model_path, image_size=args.image_size,
        min_door_size=args.min_door_size, max_door_area=args.max_door_area,
        door_erode_iterations=args.door_erode_iterations, wall_dilate=args.wall_dilate,
        door_match_radius=args.door_match_radius, min_room_size=args.min_room_size,
        max_corridor_edge_length=args.max_corridor_edge_length,
        door_merge_radius=args.door_merge_radius, room_split_h=args.room_split_h,
        spur_min_len=args.spur_min_len, device=device, mode=args.mode,
        unet_infer_size=args.unet_infer_size, stairs_merge_radius=args.stairs_merge_radius,
        corridor_merge_radius=args.corridor_merge_radius,
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
        print(f"   -> 고립 노드 {len(isolated)}개: {isolated}")

    print("결과 시각화...")
    visualize_real(mask, room_labels_only, skeleton, room_nodes, G, save_path=args.save_path)

    if result["stats"].get("mode") == "rule":
        from rule_based_floorplan import save_overlay
        overlay_path = os.path.splitext(args.save_path)[0] + "_overlay.png"
        save_overlay(args.image_path, result, overlay_path)
        print(f"오버레이 저장 완료: {os.path.abspath(overlay_path)}")

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
