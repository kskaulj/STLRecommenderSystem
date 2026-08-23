
import io

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

import config
from dataset import eval_transform
from model import CompatibilityNet

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ckpt = torch.load(config.CHECKPOINT, map_location=device, weights_only=True)
model = CompatibilityNet(_ckpt["embed_dim"], _ckpt["color_embed_dim"],
                         _ckpt["hist_bins"], pretrained=False).to(device)
model.load_state_dict(_ckpt["state_dict"])
model.eval()

_catalog = torch.load(config.CATALOG_EMB, weights_only=True)
SIGNATURES: list[str] = _catalog["signatures"]
EMBEDDINGS: torch.Tensor = _catalog["embeddings"].to(device)         
COLOR_EMBEDDINGS: torch.Tensor = _catalog["color_embeddings"].to(device)
CATEGORIES: list[str] = _catalog["categories"]
LEAF_CATEGORIES: list[str] = [c.split("|")[-1] for c in CATEGORIES]
TOP_CATEGORIES = sorted(set(LEAF_CATEGORIES))

_transform = eval_transform()


def _build_slots() -> tuple[list[dict], dict[str, str]]:

    slots, claimed, leaf_to_slot = [], set(), {}

    def add(name: str, leaves: tuple[str, ...], tier: str) -> None:
        idx = [i for i, c in enumerate(LEAF_CATEGORIES) if c in leaves]
        claimed.update(leaves)
        if not idx:                      # category absent from this catalog
            return
        slots.append({"name": name, "tier": tier,
                      "index": torch.tensor(idx, device=device)})
        leaf_to_slot.update({leaf: name for leaf in leaves})

    for name, leaves, tier in config.OUTFIT_SLOTS:
        add(name, leaves, tier)
    for leaf in TOP_CATEGORIES:
        if leaf not in claimed:
            add(leaf, (leaf,), config.OUTFIT_TIERS[-1])
    return slots, leaf_to_slot


SLOTS, LEAF_TO_SLOT = _build_slots()
SLOT_ORDER = {slot["name"]: i for i, slot in enumerate(SLOTS)}

# tells what a single dropped item is
PROBE = None
if config.CATEGORY_PROBE.exists():
    _p = torch.load(config.CATEGORY_PROBE, map_location=device, weights_only=True)
    PROBE = {
        "weight": _p["weight"].to(device), "bias": _p["bias"].to(device),
        "mean": _p["feat_mean"].to(device), "std": _p["feat_std"].to(device),
        "classes": _p["classes"],
    }
    print(f"category probe loaded (val slot accuracy on product shots: "
          f"{_p.get('val_catalog_slot_accuracy', float('nan')):.3f})")


def _tier_rank(tier: str) -> int:
    try:
        return config.OUTFIT_TIERS.index(tier)
    except ValueError:
        return config.OUTFIT_TIERS.index(config.DEFAULT_OUTFIT_TIER)


def _read_query_image():
    
    if "image" not in request.files:
        return None, (jsonify({"error": "no image uploaded"}), 400)
    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except Exception:
        return None, (jsonify({"error": "file is not a readable image"}), 400)
    return img, None


def _float_arg(name: str, default: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except ValueError:
        value = default
    return min(max(value, 0.0), 1.0)


def _embed_query(img: Image.Image, query_type: str):
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if query_type == "product":
            return model.embed_product(x)
        return model.embed_scene(x)


def _classify(img: Image.Image) -> dict | None:
    
    if PROBE is None:
        return None
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = (model.trunk_features(x) - PROBE["mean"]) / PROBE["std"]
        probs = (feat @ PROBE["weight"].T + PROBE["bias"]).softmax(dim=1).squeeze(0)
    confidence, i = probs.max(dim=0)
    category = PROBE["classes"][int(i)]
    return {"category": category, "slot": LEAF_TO_SLOT.get(category),
            "confidence": round(float(confidence), 4)}


def _blend(style: torch.Tensor, color: torch.Tensor, w: float) -> torch.Tensor:
    return (1 - w) * style + w * color


def _query_scores(q_style: torch.Tensor, q_color: torch.Tensor):
    """Per-product compatibility with the dropped image (cosine similarities)."""
    style = EMBEDDINGS @ q_style.squeeze(0)
    color = COLOR_EMBEDDINGS @ q_color.squeeze(0)
    return style, color


def _item(i: int, scores, style_scores, color_scores) -> dict:
    return {
        "signature": SIGNATURES[i],
        "score": round(scores[i].item(), 4),
        "style_score": round(style_scores[i].item(), 4),
        "color_score": round(color_scores[i].item(), 4),
        "category": CATEGORIES[i],
        "image": f"/products/{SIGNATURES[i]}.jpg",
    }


@app.get("/")
def index():
    return render_template("index.html", categories=TOP_CATEGORIES,
                           n_products=len(SIGNATURES),
                           slots=[s["name"] for s in SLOTS],
                           tiers=config.OUTFIT_TIERS,
                           default_tier=config.DEFAULT_OUTFIT_TIER,
                           cohesion=int(config.OUTFIT_COHESION_WEIGHT * 100))


@app.get("/products/<signature>.jpg")
def product_image(signature: str):
    return send_from_directory(config.PRODUCTS_DIR, f"{signature}.jpg")


@app.post("/api/recommend")
def recommend():
    img, err = _read_query_image()
    if err:
        return err

    query_type = request.form.get("query_type", "scene")
    category = request.form.get("category", "")
    top_k = int(request.form.get("top_k", 12))
    color_weight = _float_arg("color_weight", config.COLOR_SCORE_WEIGHT)

    q_style, q_color = _embed_query(img, query_type)
    style_scores, color_scores = _query_scores(q_style, q_color)
    scores = _blend(style_scores, color_scores, color_weight).cpu()
    style_scores, color_scores = style_scores.cpu(), color_scores.cpu()

    order = torch.argsort(scores, descending=True)
    results = []
    for i in order.tolist():
        if category and LEAF_CATEGORIES[i] != category:
            continue
        results.append(_item(i, scores, style_scores, color_scores))
        if len(results) >= top_k:
            break
    return jsonify({"results": results, "color_weight": color_weight})


@app.post("/api/classify")
def classify():
    
    img, err = _read_query_image()
    if err:
        return err
    detection = _classify(img)
    if detection is None:
        return jsonify({"error": "no category probe trained; "
                                 "run python train_category.py"}), 503
    detection["applied"] = bool(
        detection["slot"] and detection["confidence"] >= config.PROBE_MIN_CONFIDENCE)
    return jsonify(detection)


@app.post("/api/outfit")
def outfit():
    
    img, err = _read_query_image()
    if err:
        return err

    query_type = request.form.get("query_type", "scene")
    tier = request.form.get("tier", config.DEFAULT_OUTFIT_TIER)
    color_weight = _float_arg("color_weight", config.COLOR_SCORE_WEIGHT)
    cohesion_weight = _float_arg("cohesion_weight", config.OUTFIT_COHESION_WEIGHT)

    q_style, q_color = _embed_query(img, query_type)
    style_scores, color_scores = _query_scores(q_style, q_color)
    q_scores = _blend(style_scores, color_scores, color_weight)

    slots = [s for s in SLOTS if _tier_rank(s["tier"]) <= _tier_rank(tier)]

    
    query_slot = request.form.get("query_slot") or None
    if query_slot not in SLOT_ORDER:
        query_slot = None
    detection = None
    if query_type == "product" and query_slot is None:
        detection = _classify(img)
        if detection:
            detection["applied"] = bool(
                detection["slot"]
                and detection["confidence"] >= config.PROBE_MIN_CONFIDENCE)
            if detection["applied"]:
                query_slot = detection["slot"]
    if query_slot:
        slots = [s for s in slots if s["name"] != query_slot]

   
    slots.sort(key=lambda s: q_scores[s["index"]].max().item(), reverse=True)

    chosen: list[int] = []
    picks: list[dict] = []
    for slot in slots:
        idx = slot["index"]
        candidates = q_scores[idx]
        if chosen:
            sel = torch.tensor(chosen, device=device)
            coherence = _blend(EMBEDDINGS[idx] @ EMBEDDINGS[sel].T,
                               COLOR_EMBEDDINGS[idx] @ COLOR_EMBEDDINGS[sel].T,
                               color_weight).mean(dim=1)
            total = (1 - cohesion_weight) * candidates + cohesion_weight * coherence
        else:
            total = candidates
        i = int(idx[int(total.argmax())])
        chosen.append(i)
        picks.append({"slot": slot["name"], "index": i})

    if not picks:
        return jsonify({"error": "no catalog products available for this tier"}), 400

   
    sel = torch.tensor(chosen, device=device)
    pairwise = _blend(EMBEDDINGS[sel] @ EMBEDDINGS[sel].T,
                      COLOR_EMBEDDINGS[sel] @ COLOR_EMBEDDINGS[sel].T, color_weight)
    n = len(chosen)
    if n > 1:
        fit = (pairwise.sum(dim=1) - pairwise.diagonal()) / (n - 1)
    else:
        fit = torch.zeros(n, device=device)

    scores = q_scores.cpu()
    style_scores, color_scores = style_scores.cpu(), color_scores.cpu()
    fit = fit.cpu()
    outfit_items = []
    for rank, pick in enumerate(picks):
        item = _item(pick["index"], scores, style_scores, color_scores)
        item["slot"] = pick["slot"]
        item["fit"] = round(fit[rank].item(), 4)
        item["anchor"] = rank == 0
        outfit_items.append(item)
    outfit_items.sort(key=lambda it: SLOT_ORDER[it["slot"]])   # head-to-toe order

    query_fit = float(scores[chosen].mean())
    outfit_fit = float(fit.mean())
    return jsonify({
        "outfit": outfit_items,
        "query_slot": query_slot,
        "detection": detection,
        "color_weight": color_weight,
        "cohesion_weight": cohesion_weight,
        "outfit_score": {
            "query": round(query_fit, 4),
            "cohesion": round(outfit_fit, 4),
            "overall": round((1 - cohesion_weight) * query_fit
                             + cohesion_weight * outfit_fit, 4),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
