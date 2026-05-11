#!/usr/bin/env python3
"""
AI2-THOR demo: CLIP queries → segmentation-based highlighting → 30-second MP4.

Install extra deps (not in core requirements):
    uv pip install ai2thor opencv-python-headless

Run:
    python scripts/ai2thor_demo.py [--scene FloorPlan201] [--out demo_output/]

Output: demo_output/ai2thor_demo.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import math

import numpy as np

# ── dep checks ──────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("Missing: uv pip install opencv-python-headless")

try:
    from ai2thor.controller import Controller
except ImportError:
    sys.exit("Missing: uv pip install ai2thor")

try:
    import open_clip
    import torch
    import torch.nn.functional as F
    from PIL import Image
except ImportError:
    sys.exit("open_clip / torch already in project deps — check your venv")

# ── config ──────────────────────────────────────────────────────────────────
W, H             = 1280, 720
FPS              = 30
ENCODE_POSITIONS = 10    # positions visited during feature-extraction phase
ENCODE_YAWS      = [0, 90, 180, 270]

QUERIES: list[str] = [
    "television or screen",
    "sofa or couch",
    "floor lamp",
    "book or magazine",
    "laptop",
]

# neon yellow-green in BGR (OpenCV uses BGR)
HIGHLIGHT_BGR = (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── CLIP helpers (standalone — no ModelRegistry needed) ─────────────────────

def load_clip(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_crops(
    model, preprocess, device: torch.device,
    frame_rgb: np.ndarray,
    bboxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, np.ndarray]:
    """Encode every object crop in one batched forward pass."""
    pil = Image.fromarray(frame_rgb)
    ids, tensors = [], []
    for obj_id, (x1, y1, x2, y2) in bboxes.items():
        w, h = int(x2 - x1), int(y2 - y1)
        if w < 20 or h < 20:
            continue
        crop = pil.crop((int(x1), int(y1), int(x2), int(y2)))
        tensors.append(preprocess(crop))
        ids.append(obj_id)
    if not tensors:
        return {}
    batch = torch.stack(tensors).to(device)
    feats = model.encode_image(batch)
    feats = F.normalize(feats, dim=-1).cpu().numpy()
    return dict(zip(ids, feats))


@torch.no_grad()
def encode_texts(model, tokenizer, device: torch.device, texts: list[str]) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    feats = model.encode_text(tokens)
    return F.normalize(feats, dim=-1).cpu().numpy()


# ── AI2-THOR helpers ─────────────────────────────────────────────────────────

def teleport(controller: Controller, pos: dict, yaw: float = 0, horizon: float = 0) -> object:
    return controller.step(
        action="TeleportFull",
        **pos,
        rotation={"x": 0, "y": yaw, "z": 0},
        horizon=horizon,
        standing=True,
    )


def _bboxes_from_detections(dets: dict) -> dict[str, tuple[float, float, float, float]]:
    result = {}
    for obj_id, bb in dets.items():
        try:
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                result[obj_id] = tuple(float(x) for x in bb)  # type: ignore[assignment]
            elif isinstance(bb, dict):
                result[obj_id] = (float(bb["x1"]), float(bb["y1"]),
                                  float(bb["x2"]), float(bb["y2"]))
            elif hasattr(bb, "x1"):
                result[obj_id] = (float(bb.x1), float(bb.y1),
                                  float(bb.x2), float(bb.y2))
        except Exception:
            continue
    return result


def _bboxes_from_segmentation(event) -> dict[str, tuple[float, float, float, float]]:
    seg = getattr(event, "instance_segmentation_frame", None)
    if seg is None:
        return {}
    c2o = getattr(event, "color_to_object_id", {})
    result = {}
    for color, obj_id in c2o.items():
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        mask = (seg[:, :, 0] == r) & (seg[:, :, 1] == g) & (seg[:, :, 2] == b)
        if not mask.any():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        result[obj_id] = (float(cols[0]), float(rows[0]),
                          float(cols[-1]), float(rows[-1]))
    return result


def _bboxes_from_metadata(
    event, img_w: int, img_h: int, fov_deg: float = 60.0
) -> dict[str, tuple[float, float, float, float]]:
    """Project visible object 3D centres to screen; use estimated crop size."""
    agent = event.metadata.get("agent", {})
    pos   = agent.get("position", {})
    rot   = agent.get("rotation", {})
    cam   = np.array([pos.get("x", 0.0), pos.get("y", 0.9), pos.get("z", 0.0)])
    yaw   = math.radians(rot.get("y", 0.0))

    # AI2-THOR uses Unity left-handed coords: Y-up, Z-forward
    fwd   = np.array([ math.sin(yaw), 0.0,  math.cos(yaw)])
    right = np.array([ math.cos(yaw), 0.0, -math.sin(yaw)])
    up    = np.array([0.0, 1.0, 0.0])

    # Horizontal FOV → focal length (same for both axes, square pixels)
    fx = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = img_w / 2.0, img_h / 2.0

    result = {}
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible", False):
            continue
        p = obj.get("position", {})
        P = np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])
        d = P - cam
        pz = float(d @ fwd)
        if pz < 0.3:
            continue
        u = fx * float(d @ right) / pz + cx
        v = fx * float(-(d @ up)) / pz + cy   # screen Y is downward
        half = max(30.0, min(180.0, fx * 0.4 / pz))
        x1 = max(0.0,     u - half)
        y1 = max(0.0,     v - half)
        x2 = min(float(img_w), u + half)
        y2 = min(float(img_h), v + half)
        if x2 - x1 >= 20 and y2 - y1 >= 20:
            result[obj["objectId"]] = (x1, y1, x2, y2)
    return result


def get_bboxes(
    event, img_w: int = W, img_h: int = H
) -> dict[str, tuple[float, float, float, float]]:
    """Try three strategies in order: detections → segmentation → metadata projection."""
    dets = getattr(event, "instance_detections2D", None) or {}
    if dets:
        result = _bboxes_from_detections(dets)
        if result:
            return result

    result = _bboxes_from_segmentation(event)
    if result:
        return result

    # Fallback: project visible object centres from metadata (works on all platforms)
    return _bboxes_from_metadata(event, img_w, img_h)


def build_id_to_color(controller: Controller, positions: list[dict]) -> dict[str, tuple]:
    """Walk a few positions to collect objectId → segmentation RGB color."""
    mapping: dict[str, tuple] = {}
    for pos in positions[:4]:
        for yaw in ENCODE_YAWS:
            ev = teleport(controller, pos, yaw)
            c2o = getattr(ev, "color_to_object_id", {})
            for color, obj_id in c2o.items():
                if obj_id not in mapping:
                    mapping[obj_id] = tuple(int(c) for c in color)
    return mapping


# ── feature extraction ────────────────────────────────────────────────────────

def extract_object_features(
    controller: Controller,
    model, preprocess, device,
    positions: list[dict],
) -> dict[str, np.ndarray]:
    """Visit positions, encode visible object crops, return mean feat per objectId."""
    accum: dict[str, list[np.ndarray]] = {}
    views_with_objects = 0
    for pos in positions:
        for yaw in ENCODE_YAWS:
            ev = teleport(controller, pos, yaw)
            bboxes = get_bboxes(ev, W, H)
            if not bboxes:
                continue
            views_with_objects += 1
            feats = encode_crops(model, preprocess, device, ev.frame, bboxes)
            for obj_id, feat in feats.items():
                accum.setdefault(obj_id, []).append(feat)
    print(f"  {views_with_objects} views with detected objects")

    return {
        obj_id: np.mean(np.stack(fs), axis=0)
        for obj_id, fs in accum.items()
    }


# ── query matching ────────────────────────────────────────────────────────────

def match_queries(
    text_feats: np.ndarray,        # (Q, D)
    obj_feats: dict[str, np.ndarray],
    top_k: int = 3,
) -> list[list[str]]:
    if not obj_feats:
        return [[] for _ in text_feats]
    ids = list(obj_feats)
    img_mat = np.stack([obj_feats[i] for i in ids])   # (N, D)
    sims = text_feats @ img_mat.T                      # (Q, N)
    return [
        [ids[i] for i in np.argsort(-row)[:top_k]]
        for row in sims
    ]


# ── rendering helpers ─────────────────────────────────────────────────────────

def highlight_frame(
    frame_rgb: np.ndarray,
    seg_rgb: np.ndarray | None,
    id_to_color: dict[str, tuple],
    target_ids: set[str],
    fallback_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Blend a neon highlight onto target objects and draw their contours.

    Uses segmentation mask when available; falls back to bounding-box rectangles.
    """
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if not target_ids:
        return out

    # ── segmentation path ─────────────────────────────────────────────────
    if seg_rgb is not None and id_to_color:
        combined_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        for obj_id in target_ids:
            color = id_to_color.get(obj_id)
            if color is None:
                continue
            r, g, b = color
            mask = (
                (seg_rgb[:, :, 0] == r) &
                (seg_rgb[:, :, 1] == g) &
                (seg_rgb[:, :, 2] == b)
            ).astype(np.uint8)
            combined_mask |= mask

        if combined_mask.any():
            overlay = out.copy()
            overlay[combined_mask == 1] = HIGHLIGHT_BGR
            cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, HIGHLIGHT_BGR, 3)
            return out

    # ── bbox fallback (macOS / no segmentation) ───────────────────────────
    if fallback_bboxes:
        for obj_id in target_ids:
            bb = fallback_bboxes.get(obj_id)
            if bb is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in bb)
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), HIGHLIGHT_BGR, -1)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), HIGHLIGHT_BGR, 3)

    return out


def annotate(
    frame_bgr: np.ndarray,
    query: str,
    match_types: list[str],
    query_num: int,
    total: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    banner_h = 95

    # Dark banner at the bottom
    roi = out[H - banner_h:]
    dark = (roi * 0.35).astype(np.uint8)
    cv2.addWeighted(dark, 1.0, roi, 0.0, 0, roi)

    cv2.putText(out, f'Query: "{query}"',
                (18, H - banner_h + 32), FONT, 0.85, (255, 255, 255), 2)
    if match_types:
        cv2.putText(
            out,
            "Matched:  " + "  ·  ".join(match_types[:3]),
            (18, H - banner_h + 70),
            FONT, 0.65, (80, 230, 80), 2,
        )

    # Top-right indicator
    cv2.putText(out, f"{query_num}/{total}",
                (W - 75, 38), FONT, 0.85, (255, 255, 255), 2)
    return out


# ── interactive mode ──────────────────────────────────────────────────────────

def run_interactive(
    controller: Controller,
    model, preprocess, tokenizer, device,
    obj_feats: dict[str, np.ndarray],
    id_to_type: dict[str, str],
    id_to_color: dict[str, tuple],
    start_pos: dict,
) -> None:
    """
    Live interactive mode.

    Navigation (click the cv2 window first to capture keys):
      A / D  — rotate left / right
      W / S  — move forward / back
      Q / E  — look up / down
      Esc    — quit

    Type queries in the terminal; the window highlights the best match instantly.
    """
    import queue
    import threading

    query_queue: queue.Queue[str | None] = queue.Queue()

    def _read_queries() -> None:
        print("\nNavigation: [W/S] move  [A/D] rotate  [Q/E] look  [Esc] quit")
        print("(Click the cv2 window first so keystrokes are captured there)")
        print("─" * 60)
        while True:
            try:
                text = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                query_queue.put(None)
                return
            if text.lower() in ("quit", "exit"):
                query_queue.put(None)
                return
            if text:
                query_queue.put(text)

    threading.Thread(target=_read_queries, daemon=True).start()

    # Place agent at a known starting position
    ev = teleport(controller, start_pos, yaw=0)

    current_query      = ""
    current_target_ids: set[str] = set()
    current_matches:   list[str] = []

    def _render(event) -> np.ndarray:
        seg_rgb = getattr(event, "instance_segmentation_frame", None)
        bboxes  = get_bboxes(event, W, H)
        hi      = highlight_frame(event.frame, seg_rgb, id_to_color,
                                  current_target_ids, bboxes)
        if current_query:
            return annotate(hi, current_query, current_matches, 1, 1)
        frame = hi.copy()
        cv2.putText(frame, "Type a query in the terminal",
                    (18, 48), FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "W/S move  A/D rotate  Q/E look  Esc quit",
                    (18, 86), FONT, 0.65, (200, 200, 200), 1)
        return frame

    cv2.namedWindow("Scene Query", cv2.WINDOW_NORMAL)
    cv2.imshow("Scene Query", _render(ev))

    # Map key → (action, kwargs)
    KEY_ACTIONS: dict[int, tuple[str, dict]] = {
        ord('w'): ("MoveAhead",    {}),
        ord('s'): ("MoveBack",     {}),
        ord('a'): ("RotateLeft",   {"degrees": 30}),
        ord('d'): ("RotateRight",  {"degrees": 30}),
        ord('q'): ("LookUp",       {}),
        ord('e'): ("LookDown",     {}),
    }

    while True:
        # Check for a new query (non-blocking)
        new_ev = None
        try:
            new_q = query_queue.get_nowait()
            if new_q is None:
                break
            text_feat  = encode_texts(model, tokenizer, device, [new_q])
            matches    = match_queries(text_feat, obj_feats, top_k=1)
            best_id    = matches[0][0] if matches[0] else None
            best_type  = id_to_type.get(best_id, "") if best_id else ""
            # Highlight every object of the same type, not just the top-1
            if best_type:
                current_target_ids = {
                    oid for oid, otype in id_to_type.items() if otype == best_type
                }
            else:
                current_target_ids = set(matches[0])
            current_query   = new_q
            current_matches = [best_type] if best_type else []
            print(f"  Matched type: {best_type!r}  ({len(current_target_ids)} instance(s))")
            # Re-render current view with new highlight
            cv2.imshow("Scene Query", _render(ev))
        except queue.Empty:
            pass

        key = cv2.waitKey(50) & 0xFF
        if key == 27:   # Esc
            break

        if key in KEY_ACTIONS:
            action, kwargs = KEY_ACTIONS[key]
            ev = controller.step(action=action, **kwargs)
            cv2.imshow("Scene Query", _render(ev))

    cv2.destroyAllWindows()


# ── main ─────────────────────────────────────────────────────────────────────

def main(scene: str, out_dir: Path, interactive: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP (ViT-B/32) on {device} …")
    model, preprocess, tokenizer = load_clip(device)

    print(f"Starting AI2-THOR — {scene} …")
    controller = Controller(
        scene=scene,
        width=W,
        height=H,
        renderInstanceSegmentation=True,
        visibilityDistance=10.0,
        fieldOfView=60,
    )

    # Reachable positions
    positions: list[dict] = (
        controller.step("GetReachablePositions").metadata["actionReturn"] or []
    )
    print(f"  {len(positions)} reachable positions")
    if not positions:
        controller.stop()
        sys.exit("Scene has no reachable positions.")

    # Spread positions: take every Nth to spread across the room
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(positions), size=min(20, len(positions)), replace=False)
    spread = [positions[i] for i in sorted(idxs)]

    # Object types that are structural/background and should never be a query target
    SKIP_TYPES = {
        "Floor", "Wall", "Walls", "Ceiling", "Room",
        "Window", "Door", "Doorframe", "Doorway",
        "Blinds", "Curtains", "ShowerCurtain",
        "Painting", "Mirror", "Poster", "WallDecor",
        "RoomDecor", "Rug", "TargetCircle",
    }

    _all_objects = controller.step("Pass").metadata["objects"]
    id_to_type: dict[str, str] = {obj["objectId"]: obj["objectType"] for obj in _all_objects}
    id_to_world_pos: dict[str, dict] = {obj["objectId"]: obj["position"] for obj in _all_objects}

    # ── encode ────────────────────────────────────────────────────────────
    # Interactive mode scans fewer positions to start faster.
    encode_positions = spread[:5] if interactive else spread
    print(f"Scanning scene to build object index "
          f"({'quick pass' if interactive else 'full pass'}) …")
    obj_feats = extract_object_features(controller, model, preprocess, device, encode_positions)

    # Drop structural background objects before matching
    obj_feats = {
        oid: feat for oid, feat in obj_feats.items()
        if id_to_type.get(oid, "") not in SKIP_TYPES
    }
    print(f"  {len(obj_feats)} unique objects encoded (after filtering structural types)")

    if not obj_feats:
        controller.stop()
        sys.exit("No objects detected. Check that renderInstanceSegmentation works.")

    # ── interactive mode ──────────────────────────────────────────────────
    if interactive:
        # Skip the color-map pass — bbox fallback handles highlighting.
        run_interactive(
            controller, model, preprocess, tokenizer, device,
            obj_feats, id_to_type, id_to_color={}, start_pos=spread[0],
        )
        controller.stop()
        return

    # ── record mode ───────────────────────────────────────────────────────
    print("Building segmentation color map …")
    id_to_color = build_id_to_color(controller, spread)
    print("Matching queries …")
    text_feats = encode_texts(model, tokenizer, device, QUERIES)
    per_query_matches = match_queries(text_feats, obj_feats, top_k=1)

    # Expand each match to every instance of the winning objectType
    per_query_target_ids: list[set[str]] = []
    for q, matches in zip(QUERIES, per_query_matches):
        best_type = id_to_type.get(matches[0], "") if matches else ""
        all_ids = (
            {oid for oid, otype in id_to_type.items() if otype == best_type}
            if best_type else set(matches)
        )
        per_query_target_ids.append(all_ids)
        print(f"  \"{q}\"  →  {best_type!r}  ({len(all_ids)} instance(s))")

    # Pre-compute XZ array for all reachable positions
    pos_xyz = np.array([[p["x"], p.get("y", 0.0), p["z"]] for p in positions])

    # Candidate ranks to try per query (far first).
    # We teleport to each, count how many target objects are actually visible,
    # and keep the frame with the best coverage.
    CANDIDATE_RANKS  = [160, 180, 140, 200, 120, 100]
    HORIZON          = 15    # degrees: tilt camera slightly down for room coverage
    SECS_PER_SNIPPET = 3     # 5 queries × 3 s = 15 s total
    FRAMES_PER_SNIPPET = FPS * SECS_PER_SNIPPET

    out_path = str(out_dir / "ai2thor_demo.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, float(FPS), (W, H))

    total_queries = len(QUERIES)
    total_secs    = total_queries * SECS_PER_SNIPPET
    print(f"Recording ~{total_secs}s at {FPS}fps → {out_path}")

    for q_idx, (query, target_ids) in enumerate(zip(QUERIES, per_query_target_ids)):
        best_type   = next((id_to_type.get(m, "") for m in target_ids if m in id_to_type), "")
        match_types = [best_type] if best_type else []
        print(f"  [{q_idx + 1}/{total_queries}] \"{query}\"  ({len(target_ids)} instance(s))")

        # Centroid of all matched objects
        obj_positions = [id_to_world_pos[oid] for oid in target_ids if oid in id_to_world_pos]
        centroid = np.array([
            np.mean([p["x"] for p in obj_positions]),
            np.mean([p["y"] for p in obj_positions]),
            np.mean([p["z"] for p in obj_positions]),
        ]) if obj_positions else pos_xyz[0]

        # Sort reachable positions by distance to centroid
        dists = np.linalg.norm(pos_xyz - centroid, axis=1)
        order = np.argsort(dists)   # closest first

        # Pick the candidate position where the most target objects are visible
        best_ev, best_count = None, -1
        for rank in CANDIDATE_RANKS:
            idx = order[min(rank, len(order) - 1)]
            wp  = positions[idx]
            dx  = float(centroid[0] - wp["x"])
            dz  = float(centroid[2] - wp["z"])
            yaw = math.degrees(math.atan2(dx, dz))
            ev  = teleport(controller, wp, yaw, horizon=HORIZON)

            visible_ids = {
                obj["objectId"] for obj in ev.metadata.get("objects", [])
                if obj.get("visible")
            }
            count = len(target_ids & visible_ids)
            if count > best_count:
                best_count, best_ev = count, ev
            if best_count == len(target_ids):
                break   # all targets visible — no need to try more

        ev = best_ev or ev
        seg_rgb = getattr(ev, "instance_segmentation_frame", None)
        for color, obj_id in getattr(ev, "color_to_object_id", {}).items():
            if obj_id not in id_to_color:
                id_to_color[obj_id] = tuple(int(c) for c in color)

        bboxes      = get_bboxes(ev, W, H)
        highlighted = highlight_frame(ev.frame, seg_rgb, id_to_color, target_ids, bboxes)
        frame       = annotate(highlighted, query, match_types, q_idx + 1, total_queries)
        print(f"     {best_count}/{len(target_ids)} target(s) visible")

        for _ in range(FRAMES_PER_SNIPPET):
            writer.write(frame)

    writer.release()
    controller.stop()

    duration = total_queries * SECS_PER_SNIPPET
    size_mb  = Path(out_path).stat().st_size / 1e6
    print(f"\nDone — {out_path}  ({duration}s, {size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI2-THOR CLIP query demo")
    parser.add_argument("--scene",  default="FloorPlan201",
                        help="AI2-THOR scene name (default: FloorPlan201 — living room)")
    parser.add_argument("--out",    default="demo_output",
                        help="Output directory for recorded video (default: demo_output/)")
    parser.add_argument("--interactive", action="store_true",
                        help="Open a live window — type queries in the terminal instead of recording")
    args = parser.parse_args()
    main(args.scene, Path(args.out), interactive=args.interactive)
