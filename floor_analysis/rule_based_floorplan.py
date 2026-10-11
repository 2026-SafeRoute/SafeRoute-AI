"""
규칙 기반 도면 분석 (U-Net 없이 벽선을 직접 읽는 방식)

흰 배경에 검은 벽선이 깨끗하게 그려진 도면이 대상.
학습 데이터가 적어 U-Net이 이런 도면의 벽을 제대로 예측하지 못하는 문제를
피하기 위해, 벽을 "예측"하지 않고 이미지에서 그대로 읽는다.

처리 순서:
  1) 굵은 검은 선 = 벽. 글자/기호(작은 덩어리)와 얇은 선(문 호, 계단 빗금)은 제외
  2) 건물 바깥 제외 (테두리와 이어진 빈 공간)
  3) 내부 빈 공간을 distance transform + watershed로 분할 -> 좁은 목(문 틈)에서 나뉨
  4) 넓게 맞닿은 영역끼리는 다시 합침 (문이 아니라 그냥 트인 공간)
  5) 남은 접촉 구간 하나하나 = 문
  6) 문이 많은 영역 = 복도, 빗금이 많은 영역 = 계단, 나머지 = 방
  7) 방/계단은 중심 노드 1개, 복도는 중심선을 따라 일정 간격으로 노드 생성

analyze_floorplan_rule_based()는 run_real_floorplan_pipeline.analyze_floorplan()과
같은 형식(dict: G, mask, room_labels_only, skeleton, room_nodes, stats)으로 반환하고,
노드 좌표도 같은 기준(image_size x image_size 정사각)으로 맞춘다.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import h_maxima, skeletonize
from skimage.segmentation import watershed
import networkx as nx

import test_pipeline as tp

EIGHT = np.ones((3, 3), dtype=int)


def _load_rgb(image_path):
    """투명 배경(RGBA)은 흰색 위에 합성해서 읽음."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.array(img.convert("RGB"))


def _small_blobs(dark, max_dim):
    """글자/기호처럼 벽과 떨어져 있는 작은 덩어리 마스크와 그 중심 좌표들."""
    lab, _ = ndimage.label(dark, structure=EIGHT)
    blobs = np.zeros_like(dark)
    centers = []
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        bh, bw = slc[0].stop - slc[0].start, slc[1].stop - slc[1].start
        if max(bh, bw) < max_dim:
            blobs[slc][lab[slc] == i] = True
            centers.append(((slc[0].start + slc[0].stop) / 2, (slc[1].start + slc[1].stop) / 2))
    return blobs, centers


def _number_label_centers(text, w, labels):
    """
    방 번호처럼 글자 여러 개가 가로로 붙어 있는 묶음의 중심 좌표.
    글자 하나짜리(기호, 아이콘)는 제외. 벽 너머 옆방 번호와 한 묶음이 되지 않도록
    같은 영역(labels) 안의 글자끼리만 묶음.
    """
    glyphs, n = ndimage.label(text, structure=EIGHT)
    if n == 0:
        return []
    reach = max(2, round(0.008 * w))
    group, _ = ndimage.label(ndimage.binary_dilation(text, structure=np.ones((1, 2 * reach + 1))),
                             structure=EIGHT)
    members = {}
    for cy, cx in ndimage.center_of_mass(text, glyphs, range(1, n + 1)):
        cy, cx = int(cy), int(cx)
        members.setdefault((int(group[cy, cx]), int(labels[cy, cx])), []).append((cy, cx))
    return [tuple(np.mean(pts, axis=0).astype(int)) for (g, host), pts in members.items()
            if host > 0 and len(pts) >= 2]


def _hatch_mask(lines, w, min_transitions=6):
    """
    계단 빗금처럼 평행선이 촘촘한 구역. 선 굵기와 무관하게, 일정 길이 구간 안에서
    선을 몇 번 넘나드는지(전환 횟수)로 판정.
    """
    L = max(9, round(0.03 * w))
    side = max(3, L // 3)
    v = np.abs(np.diff(lines.astype(np.int8), axis=0, prepend=0)).astype(np.float32)
    hz = np.abs(np.diff(lines.astype(np.int8), axis=1, prepend=0)).astype(np.float32)
    count_v = ndimage.uniform_filter(v, size=(L, side)) * L
    count_h = ndimage.uniform_filter(hz, size=(side, L)) * L
    hatch = (count_v >= min_transitions) | (count_h >= min_transitions)
    hatch = ndimage.binary_opening(hatch, structure=EIGHT)

    lab, n = ndimage.label(hatch)
    if n:
        sizes = ndimage.sum(hatch, lab, index=range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= (0.02 * w) ** 2
        hatch = keep[lab]
    return hatch


def _find_contacts(labels):
    """서로 다른 영역이 직접 맞닿은 구간들. 구간 하나 = {pair, pts, length}."""
    h, w = labels.shape
    n = int(labels.max()) + 1
    key = np.zeros((h, w), dtype=np.int64)
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: h - dy, : w - dx]
        b = labels[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        key[: h - dy, : w - dx][m] = (lo * n + hi)[m]

    comp, _ = ndimage.label(key > 0, structure=EIGHT)
    segments = []
    for i, slc in enumerate(ndimage.find_objects(comp), start=1):
        sub = comp[slc] == i
        keys = key[slc][sub]
        ys, xs = np.nonzero(sub)
        for kv in np.unique(keys):
            sel = keys == kv
            pts = np.stack([ys[sel] + slc[0].start, xs[sel] + slc[1].start], axis=1)
            segments.append({"pair": (int(kv // n), int(kv % n)), "pts": pts, "length": len(pts)})
    return segments


def _merge_regions(labels, segments, region_width, open_ratio, min_area, protected, locked=()):
    """
    접촉 폭이 영역 자체의 폭에 가까우면(문이 아니라 그냥 트인 공간) 합치고,
    너무 작은 조각은 가장 길게 맞닿은 이웃에 흡수. 이웃이 없는 작은 조각은 버림.
    문은 방보다 훨씬 좁고, 복도가 꺾이거나 넓어지는 지점은 복도 폭만큼 넓다는 점을 이용.
    """
    n = int(labels.max()) + 1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in segments:
        p, q = s["pair"]
        narrow_id = p if region_width[p] <= region_width[q] else q
        narrower = region_width[narrow_id]
        if protected[narrow_id] or p in locked or q in locked:
            continue  # 번호가 적힌 방: 문이 방 폭만큼 넓어도 복도와 합치지 않음
        if narrower > 0 and s["length"] / narrower >= open_ratio:
            a, b = find(s["pair"][0]), find(s["pair"][1])
            if a != b:
                parent[b] = a

    areas = np.bincount(labels.ravel(), minlength=n).astype(np.int64)
    root_area = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        root_area[find(i)] += areas[i]

    touch = {}
    for s in segments:
        a, b = find(s["pair"][0]), find(s["pair"][1])
        if a == b:
            continue
        touch[(a, b)] = touch.get((a, b), 0) + s["length"]
        touch[(b, a)] = touch.get((b, a), 0) + s["length"]

    removed = set()
    small_roots = sorted((r for r in range(1, n) if find(r) == r and 0 < root_area[r] < min_area),
                         key=lambda r: root_area[r])
    for r in small_roots:
        r = find(r)
        if root_area[r] >= min_area:
            continue
        neighbors = {}
        for (a, b), length in touch.items():
            if find(a) == r and find(b) != r:
                neighbors[find(b)] = neighbors.get(find(b), 0) + length
        if not neighbors:
            removed.add(r)
            continue
        target = max(neighbors, key=neighbors.get)
        parent[r] = target
        root_area[target] += root_area[r]

    lut = np.zeros(n, dtype=np.int32)
    new_ids = {}
    for i in range(1, n):
        r = find(i)
        if r in removed or areas[i] == 0:
            continue
        if r not in new_ids:
            new_ids[r] = len(new_ids) + 1
        lut[i] = new_ids[r]
    return lut[labels]


def _prune_leaf_chains(G, min_len):
    """끝점에서 분기점까지의 짧은 가지를 한 번만 제거 (연쇄적으로 깎아먹지 않도록 단일 패스)."""
    to_remove = []
    for leaf in [n for n, d in G.degree() if d == 1]:
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
            to_remove.extend(path[:-1])
    G.remove_nodes_from(to_remove)


def _corridor_centerline_nodes(region, dist, spacing):
    """
    복도 영역의 중심선을 따라 일정 간격으로 노드를 배치.
    반환: (nodes: [(row, col)], edges: [(i, j)], skeleton_pixels: (N,2))
    """
    sk = skeletonize(region)
    if sk.sum() < 2:
        r, c = np.unravel_index(np.argmax(np.where(region, dist, -1)), region.shape)
        return [(float(r), float(c))], [], np.argwhere(sk)

    raw = tp.skeleton_to_graph(sk)
    half_width = float(np.median(dist[sk]))
    _prune_leaf_chains(raw, min_len=2.2 * half_width)

    keys = {n for n, d in raw.degree() if d != 2}
    if not keys:  # 분기도 끝점도 없는 순환 고리
        keys = {next(iter(raw.nodes))}

    nodes, edges = [], []
    cluster_of = {}
    for cluster in nx.connected_components(raw.subgraph(keys)):
        pts = np.array(list(cluster), dtype=float)
        center = pts.mean(axis=0)
        r, c = pts[np.argmin(((pts - center) ** 2).sum(axis=1))]  # 중심선 위 픽셀로 스냅
        idx = len(nodes)
        nodes.append((float(r), float(c)))
        for p in cluster:
            cluster_of[p] = idx

    chains = raw.subgraph(set(raw.nodes) - keys)
    for comp in nx.connected_components(chains):
        sub = chains.subgraph(comp)
        ends = [n for n in comp if sub.degree(n) <= 1]
        order = list(nx.dfs_preorder_nodes(sub, ends[0])) if len(comp) > 1 else list(comp)

        start_keys = [k for k in raw.neighbors(order[0]) if k in keys]
        end_keys = [k for k in raw.neighbors(order[-1]) if k in keys]
        if not start_keys or not end_keys:
            continue
        a = cluster_of[start_keys[0]]
        b = cluster_of[end_keys[-1]]

        path = [nodes[a]] + [(float(r), float(c)) for r, c in order] + [nodes[b]]
        steps = [np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 for i in range(len(path) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(steps)])
        total = cum[-1]
        n_seg = max(1, int(round(total / spacing)))

        prev = a
        for i in range(1, n_seg):
            j = int(np.searchsorted(cum, total * i / n_seg))
            idx = len(nodes)
            nodes.append(path[min(j, len(path) - 1)])
            edges.append((prev, idx))
            prev = idx
        if prev != b:
            edges.append((prev, b))

    kept = np.array(list(raw.nodes)) if raw.number_of_nodes() else np.zeros((0, 2), dtype=int)
    return nodes, edges, kept


def analyze_floorplan_rule_based(
    image_path, image_size=512, dark_thresh=128, wall_open=0,
open_merge_ratio=0.65, corridor_min_doors=4, corridor_node_spacing=30,
    min_region_frac=0.0004, stairs_min_frac=0.25, exterior_max_frac=0.15, hatch_contrast=18, max_side=2400,
):
    """
    wall_open: 벽으로 인정할 최소 선 굵기(픽셀). 0이면 이미지 폭에 맞춰 자동.
    open_merge_ratio: 접촉 폭이 (더 좁은 쪽 영역 폭 x 이 값) 이상이면 문이 아니라 트인 공간으로 보고 합침.
    corridor_min_doors: 문이 이 개수 이상 달린 영역을 복도로 판정.
    corridor_node_spacing: 복도 노드 간격 (image_size 기준 픽셀).
    max_side: 이미지의 긴 변이 이보다 크면 줄여서 분석 (처리 시간 제한용).
              노드 좌표는 image_size 기준으로 정규화되므로 결과 좌표에는 영향 없음.
    """
    rgb = _load_rgb(image_path)
    if max_side and max(rgb.shape[:2]) > max_side:
        ratio = max_side / max(rgb.shape[:2])
        new_size = (max(1, round(rgb.shape[1] * ratio)), max(1, round(rgb.shape[0] * ratio)))
        rgb = np.array(Image.fromarray(rgb).resize(new_size, Image.LANCZOS))
    h, w = rgb.shape[:2]
    scale_r, scale_c = image_size / h, image_size / w
    gray = rgb.astype(np.int16).mean(axis=2)
    dark = gray < dark_thresh

    # ---- 1) 벽 추출 ----
    text, text_centers = _small_blobs(dark, max_dim=0.04 * w)
    lines = dark & ~text
    # 계단 빗금은 분홍/회색처럼 연한 선으로 그려진 경우가 있어서, 절대 밝기가 아니라
    # 주변 평균보다 얼마나 어두운지로 선을 잡음 (바닥색이 칠해진 방은 대비가 없어 걸리지 않음)
    win = max(5, round(0.012 * w)) | 1
    local_dark = (ndimage.uniform_filter(gray.astype(np.float32), size=win) - gray) > hatch_contrast
    hatch = _hatch_mask(local_dark & ~text, w)

    k = wall_open or max(2, round(w / 400))
    wall = ndimage.binary_opening(lines, structure=np.ones((k, k)))
    wall_lab, _ = ndimage.label(wall, structure=EIGHT)
    for i, slc in enumerate(ndimage.find_objects(wall_lab), start=1):
        if max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start) < 0.04 * w:
            wall[slc][wall_lab[slc] == i] = False

    # ---- 2) 건물 바깥 제외 (얇은 선까지 포함해서 판정: 넓은 외부 출입문으로 새는 것 방지) ----
    closed_lines = ndimage.binary_closing(dark, structure=EIGHT)
    out_lab, _ = ndimage.label(~closed_lines)
    border = np.unique(np.concatenate([out_lab[0], out_lab[-1], out_lab[:, 0], out_lab[:, -1]]))
    outside = np.isin(out_lab, border[border > 0])
    outside = ndimage.binary_dilation(outside, iterations=2)
    free = ~wall & ~outside

    # ---- 3) watershed 분할 ----
    dist = ndimage.distance_transform_edt(free)
    markers, _ = ndimage.label(h_maxima(dist, max(2.0, 0.004 * w)))
    labels = watershed(-dist, markers, mask=free).astype(np.int32)

    # ---- 3-1) 복도 쪽이 넓게 트인 방 살리기 ----
    # 벽 모양만으로는 복도와 구분되지 않는 방이 있음. 문이 많이 달린 영역(복도) 안에
    # 방 번호가 있으면, 그 근처에서 벽에서 가장 먼 지점을 씨앗으로 추가해서 다시 분할.
    segments = _find_contacts(labels)
    contact_count = np.zeros(int(labels.max()) + 1, dtype=int)
    for s in segments:
        if s["length"] >= 3:
            contact_count[s["pair"][0]] += 1
            contact_count[s["pair"][1]] += 1
    rescue_points = []
    radius = max(4, round(0.02 * w))
    for cy, cx in _number_label_centers(text, w, labels):
        host = labels[cy, cx]
        if host == 0 or contact_count[host] < corridor_min_doors:
            continue
        ys = slice(max(0, cy - radius), min(h, cy + radius + 1))
        xs = slice(max(0, cx - radius), min(w, cx + radius + 1))
        local = np.where(labels[ys, xs] == host, dist[ys, xs], -1)
        r, c = np.unravel_index(np.argmax(local), local.shape)
        rescue_points.append((r + ys.start, c + xs.start))
    if rescue_points:
        next_marker = int(markers.max())
        for r, c in rescue_points:
            if markers[r, c] == 0:
                next_marker += 1
                markers[r, c] = next_marker
        labels = watershed(-dist, markers, mask=free).astype(np.int32)
        segments = _find_contacts(labels)
    locked = {int(labels[r, c]) for r, c in rescue_points}

    # ---- 4) 트인 공간 병합 + 작은 조각 정리 ----
    n_labels = int(labels.max())
    region_width = np.zeros(n_labels + 1)
    if n_labels:
        region_width[1:] = 2 * ndimage.maximum(dist, labels, index=range(1, n_labels + 1))
    pre_merge_contacts = [
        (int(s["pts"][:, 1].mean()), int(s["pts"][:, 0].mean()), s["length"],
         round(s["length"] / max(1e-6, min(region_width[s["pair"][0]], region_width[s["pair"][1]])), 2))
        for s in segments
    ]
    # 이웃이 하나뿐(막다른 곳)이고 안에 글자가 있는 영역은 번호가 적힌 방으로 보고 보호
    neighbors = [set() for _ in range(n_labels + 1)]
    for s in segments:
        a, b = s["pair"]
        neighbors[a].add(b)
        neighbors[b].add(a)
    blobs_in = np.zeros(n_labels + 1, dtype=int)
    for cy, cx in text_centers:
        blobs_in[labels[min(h - 1, int(cy)), min(w - 1, int(cx))]] += 1
    protected = np.array([len(neighbors[i]) == 1 and blobs_in[i] >= 1 for i in range(n_labels + 1)])
    labels = _merge_regions(labels, segments, region_width, open_merge_ratio,
                            min_area=min_region_frac * h * w, protected=protected, locked=locked)

    # ---- 건물 바깥에 붙은 영역(현관 포치, 마당 등) 제외 ----
    # 벽은 넘지 못하게 하고(mask) 영역을 조금 넓혀서 바깥과 닿는 부분을 찾음.
    # 테두리의 상당 부분이 바깥과 닿으면 건물 밖 영역, 일부만 닿으면 그 지점이 출구.
    dropped_exterior = 0
    exits = []  # (영역 id, 출구 픽셀 좌표들)
    pad = 4
    for rid, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        ys = slice(max(0, slc[0].start - pad), min(h, slc[0].stop + pad))
        xs = slice(max(0, slc[1].start - pad), min(w, slc[1].stop + pad))
        region = labels[ys, xs] == rid
        grown = ndimage.binary_dilation(region, iterations=pad, mask=~wall[ys, xs])
        touch = grown & outside[ys, xs]
        perimeter = (ndimage.binary_dilation(region) & ~region).sum()
        if perimeter and touch.sum() / (perimeter * pad) > exterior_max_frac:
            labels[ys, xs][region] = 0
            dropped_exterior += 1
            continue
        touch_lab, n_touch = ndimage.label(touch, structure=EIGHT)
        for t in range(1, n_touch + 1):
            pts = np.argwhere(touch_lab == t)
            if len(pts) >= 2 * pad:
                exits.append((rid, pts + np.array([ys.start, xs.start])))

    # ---- 5) 문 = 남은 접촉 구간 ----
    contacts = _find_contacts(labels)
    lengths = [s["length"] for s in contacts if s["length"] >= 3]
    door_width = float(np.median(lengths)) if lengths else 0.0
    min_door_len = max(3.0, 0.3 * door_width)
    doors = [s for s in contacts if s["length"] >= min_door_len]

    region_ids = [int(i) for i in np.unique(labels) if i > 0]
    door_count = {rid: 0 for rid in region_ids}
    for s in doors:
        for rid in s["pair"]:
            door_count[rid] += 1

    # ---- 6) 영역 분류 ----
    kind, center, area, label_blobs, hatch_frac = {}, {}, {}, {}, {}
    objects = ndimage.find_objects(labels)
    for rid in region_ids:
        slc = objects[rid - 1]
        region = labels[slc] == rid
        area[rid] = int(region.sum())
        r, c = np.unravel_index(np.argmax(np.where(region, dist[slc], -1)), region.shape)
        center[rid] = (r + slc[0].start, c + slc[1].start)
        label_blobs[rid] = sum(
            1 for cy, cx in text_centers
            if labels[min(h - 1, int(cy)), min(w - 1, int(cx))] == rid
        )
        hatch_frac[rid] = float((hatch[slc] & region).sum() / area[rid])
        if hatch_frac[rid] >= stairs_min_frac:
            kind[rid] = "stairs"
        elif door_count[rid] >= corridor_min_doors:
            kind[rid] = "corridor"
        else:
            kind[rid] = "room"

    if region_ids and "corridor" not in kind.values():
        # 문 개수 기준을 넘는 영역이 없으면, 문이 가장 많은 영역을 복도로 봄
        best = max((r for r in region_ids if kind[r] == "room"), key=lambda r: door_count[r], default=None)
        if best is not None and door_count[best] >= 2:
            kind[best] = "corridor"

    # 복도와 문보다 훨씬 넓게 트여 이어진 통과 공간(문 2개 이상)은 복도의 연장으로 봄.
    # 홀이 꺾이면서 폭이 달라지면 한 덩어리로 합쳐지지 않고 '방'으로 남는 경우를 잡기 위함.
    final_width = {rid: 2 * float(dist[objects[rid - 1]][labels[objects[rid - 1]] == rid].max())
                   for rid in region_ids}
    changed = True
    while changed:
        changed = False
        for s in doors:
            a, b = s["pair"]
            # 보통 문보다 2배 이상 넓고, 좁은 쪽 영역 폭의 35% 이상을 차지하는 트임만 해당
            if s["length"] < 2 * door_width or s["length"] < 0.35 * min(final_width[a], final_width[b]):
                continue
            for corridor_side, other in ((a, b), (b, a)):
                if kind[corridor_side] == "corridor" and kind[other] == "room" and door_count[other] >= 2:
                    kind[other] = "corridor"
                    changed = True

    # ---- 7) 그래프 구성 ----
    def scaled(r, c):
        return float(r * scale_r), float(c * scale_c)

    def add_edge(G, u, v):
        dr = G.nodes[u]["row"] - G.nodes[v]["row"]
        dc = G.nodes[u]["col"] - G.nodes[v]["col"]
        G.add_edge(u, v, weight=float(np.hypot(dr, dc)))

    G = nx.Graph()
    spacing_native = corridor_node_spacing / ((scale_r + scale_c) / 2)
    region_node = {}          # 방/계단 영역 -> 노드 id
    corridor_nodes = {}       # 복도 영역 -> [(node id, 원본 row, 원본 col)]
    skeleton_native = []
    room_nodes = {}
    room_label_map = np.zeros((h, w), dtype=np.int32)
    next_room_id = 1

    for rid in region_ids:
        if kind[rid] == "corridor":
            slc = objects[rid - 1]
            region = labels[slc] == rid
            pts, edges, sk_pts = _corridor_centerline_nodes(region, dist[slc], spacing_native)
            ids = []
            for i, (r, c) in enumerate(pts):
                nr, nc = r + slc[0].start, c + slc[1].start
                node_id = f"corridor_{rid}_{i}"
                row, col = scaled(nr, nc)
                G.add_node(node_id, row=row, col=col, room_id=None,
                           space_type="corridor", is_door=False, is_exit=False)
                ids.append((node_id, nr, nc))
            for i, j in edges:
                add_edge(G, ids[i][0], ids[j][0])
            corridor_nodes[rid] = ids
            if len(sk_pts):
                skeleton_native.append(sk_pts + np.array([slc[0].start, slc[1].start]))
        else:
            node_id = f"{kind[rid]}_{next_room_id}"
            row, col = scaled(*center[rid])
            G.add_node(node_id, row=row, col=col, room_id=next_room_id, space_type=kind[rid],
                       is_door=False, is_exit=False, has_label=label_blobs[rid] >= 2)
            region_node[rid] = node_id
            room_nodes[next_room_id] = {"row": row, "col": col, "space_type": kind[rid],
                                        "area": int(area[rid] * scale_r * scale_c)}
            room_label_map[labels == rid] = next_room_id
            next_room_id += 1

    def attach_point(rid, dr, dc):
        """문에서 그 영역 쪽으로 연결할 노드: 방/계단은 중심 노드, 복도는 가장 가까운 중심선 노드."""
        if rid in region_node:
            return region_node[rid]
        ids = corridor_nodes.get(rid)
        if not ids:
            return None
        return min(ids, key=lambda t: (t[1] - dr) ** 2 + (t[2] - dc) ** 2)[0]

    door_mask = np.zeros((h, w), dtype=bool)
    for i, s in enumerate(doors, start=1):
        dr, dc = s["pts"].mean(axis=0)
        door_mask[s["pts"][:, 0], s["pts"][:, 1]] = True
        row, col = scaled(dr, dc)
        if all(kind[rid] == "corridor" for rid in s["pair"]):
            # 복도와 복도가 만나는 트인 지점은 문이 아니라 복도 노드로 이어 줌
            node_id = f"corridor_link_{i}"
            G.add_node(node_id, row=row, col=col, room_id=None, space_type="corridor",
                       is_door=False, is_exit=False)
        else:
            node_id = f"door_{i}"
            G.add_node(node_id, row=row, col=col, room_id=None, space_type="door",
                       is_door=True, is_exit=False)
        for rid in s["pair"]:
            target = attach_point(rid, dr, dc)
            if target is not None:
                add_edge(G, node_id, target)

    for i, (rid, pts) in enumerate(exits, start=1):
        if rid not in kind:
            continue
        er, ec = pts.mean(axis=0)
        door_mask[pts[:, 0], pts[:, 1]] = True
        node_id = f"exit_{i}"
        row, col = scaled(er, ec)
        G.add_node(node_id, row=row, col=col, room_id=None, space_type="door",
                   is_door=True, is_exit=True)
        target = attach_point(rid, er, ec)
        if target is not None:
            add_edge(G, node_id, target)

    # ---- 기존 파이프라인과 같은 형식의 출력 배열 (image_size 기준) ----
    def to_square(arr):
        return np.array(Image.fromarray(arr).resize((image_size, image_size), Image.NEAREST))

    mask_native = np.full((h, w), tp.WALL, dtype=np.uint8)
    mask_native[labels > 0] = tp.FLOOR
    mask_native[ndimage.binary_dilation(door_mask, iterations=max(1, k))] = tp.DOOR

    skeleton = np.zeros((image_size, image_size), dtype=bool)
    if skeleton_native:
        pts = np.concatenate(skeleton_native)
        rr = np.clip((pts[:, 0] * scale_r).round().astype(int), 0, image_size - 1)
        cc = np.clip((pts[:, 1] * scale_c).round().astype(int), 0, image_size - 1)
        skeleton[rr, cc] = True

    kinds = list(kind.values())
    num_door_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("is_door") and not d.get("is_exit"))
    return {
        "G": G,
        "mask": to_square(mask_native),
        "room_labels_only": to_square(room_label_map).astype(np.int32),
        "skeleton": skeleton,
        "room_nodes": room_nodes,
        "stats": {
            "mode": "rule",
            # 기존 U-Net 방식의 stats 항목 이름도 유지 (이 값을 읽는 쪽이 깨지지 않도록)
            "total_doors": num_door_nodes,
            "removed_doors": 0,
            "shrunk_doors": 0,
            "removed_small_rooms": 0,
            "merged_door_nodes": 0,
            "merged_stairs_nodes": 0,
            "num_rooms": kinds.count("room"),
            "num_labeled_rooms": sum(1 for r in region_ids if kind[r] == "room" and label_blobs[r] >= 2),
            "num_corridor_regions": kinds.count("corridor"),
            "num_stairs": kinds.count("stairs"),
            "num_doors": num_door_nodes,
            "rooms_split_by_number": len(rescue_points),
            "num_exits": sum(1 for _, d in G.nodes(data=True) if d.get("is_exit")),
            "door_width_px": round(door_width, 1),
            "dropped_exterior_regions": dropped_exterior,
            "isolated_nodes": sum(1 for _ in nx.isolates(G)),
        },
        "native": {
            "labels": labels, "kind": kind, "shape": (h, w),
            "pre_merge_contacts": pre_merge_contacts,  # (x, y, 접촉 폭, 영역 폭 대비 비율)
            "regions": {rid: {"kind": kind[rid], "doors": door_count[rid], "area": area[rid],
                              "hatch_frac": round(hatch_frac[rid], 3), "center": center[rid],
                              "label_blobs": label_blobs[rid]} for rid in region_ids},
        },
    }


def save_overlay(image_path, result, save_path):
    """원본 도면 위에 영역/노드/엣지를 그려서 눈으로 검토하기 위한 그림."""
    labels, kind = result["native"]["labels"], result["native"]["kind"]
    h, w = result["native"]["shape"]
    rgb = _load_rgb(image_path)
    if rgb.shape[:2] != (h, w):  # 분석 때 max_side로 줄인 경우 같은 크기로 맞춤
        rgb = np.array(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))
    rgb = rgb.astype(np.float32)
    G = result["G"]
    size = result["mask"].shape[0]

    rng = np.random.RandomState(1)
    fixed = {"corridor": np.array([205, 110, 215]), "stairs": np.array([240, 200, 60])}
    for rid, k in kind.items():
        color = fixed.get(k, rng.randint(90, 230, 3))
        m = labels == rid
        rgb[m] = 0.5 * rgb[m] + 0.5 * color

    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    def xy(n):
        return G.nodes[n]["col"] * w / size, G.nodes[n]["row"] * h / size

    for u, v in G.edges:
        draw.line([xy(u), xy(v)], fill=(70, 70, 70), width=max(1, w // 700))
    r = max(3, w // 260)
    node_color = {"room": (30, 140, 40), "corridor": (120, 40, 170), "stairs": (200, 120, 0)}
    for n, d in G.nodes(data=True):
        x, y = xy(n)
        if d.get("is_exit"):
            draw.rectangle([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(230, 30, 30), outline=(0, 0, 0))
        elif d.get("is_door"):
            draw.rectangle([x - r, y - r, x + r, y + r], fill=(255, 140, 0), outline=(0, 0, 0))
        else:
            rr = r if d["space_type"] == "corridor" else r + 2
            draw.ellipse([x - rr, y - rr, x + rr, y + rr],
                         fill=node_color.get(d["space_type"], (120, 120, 120)), outline=(0, 0, 0))
    img.save(save_path)
