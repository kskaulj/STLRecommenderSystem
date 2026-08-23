# the actual jpges live on Pinterests CDN.

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

import config


def signature_to_url(signature: str) -> str:
    return "https://i.pinimg.com/400x/{}/{}/{}/{}.jpg".format(
        signature[0:2], signature[2:4], signature[4:6], signature
    )


def download_one(session: requests.Session, signature: str, dest_dir: Path) -> bool:
    dest = dest_dir / f"{signature}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = session.get(signature_to_url(signature), timeout=20)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            dest.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=config.NUM_PAIRS)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config.SCENES_DIR.mkdir(parents=True, exist_ok=True)
    config.PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.FASHION_JSON, encoding="utf-8") as f:
        all_pairs = [json.loads(line) for line in f if line.strip()]
    print(f"{len(all_pairs)} pairs in fashion.json")

    random.seed(config.SEED)
    sampled = random.sample(all_pairs, min(args.pairs, len(all_pairs)))

    jobs = {}  # (kind, signature) -> dest dir
    for p in sampled:
        jobs[("scene", p["scene"])] = config.SCENES_DIR
        jobs[("product", p["product"])] = config.PRODUCTS_DIR
    print(f"{len(sampled)} pairs sampled -> {len(jobs)} unique images to fetch")

    ok: set[tuple[str, str]] = set()
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research dataset download)"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, session, sig, dest): (kind, sig)
            for (kind, sig), dest in jobs.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            if fut.result():
                ok.add(futures[fut])

    kept = [
        p for p in sampled
        if ("scene", p["scene"]) in ok and ("product", p["product"]) in ok
    ]
    with open(config.PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f)

    n_products = len({p["product"] for p in kept})
    n_scenes = len({p["scene"] for p in kept})
    print(f"downloaded ok: {len(ok)}/{len(jobs)} images")
    print(f"kept {len(kept)} pairs ({n_scenes} scenes, {n_products} products)")
    print(f"wrote {config.PAIRS_FILE}")


if __name__ == "__main__":
    main()
