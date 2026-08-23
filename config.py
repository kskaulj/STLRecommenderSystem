
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent          # folder with fashion.json etc.

FASHION_JSON = DATASET_DIR / "fashion.json"
FASHION_CAT_JSON = DATASET_DIR / "fashion-cat.json"

IMAGES_DIR = PROJECT_DIR / "images"
SCENES_DIR = IMAGES_DIR / "scenes"
PRODUCTS_DIR = IMAGES_DIR / "products"

DATA_DIR = PROJECT_DIR / "data"
PAIRS_FILE = DATA_DIR / "pairs_downloaded.json"   # pairs whose images exist locally

MODELS_DIR = PROJECT_DIR / "models"
CHECKPOINT = MODELS_DIR / "compat_net.pt"
CATALOG_EMB = MODELS_DIR / "catalog_embeddings.pt"
CATEGORY_PROBE = MODELS_DIR / "category_probe.pt"

# data
NUM_PAIRS = 72198         # scene-product pairs sampled from fashion.json
                        
                          
SEED = 42

# model/training
EMBED_DIM = 128           # style-space dimension (paper uses d=128)
COLOR_EMBED_DIM = 64      # color/pattern-space dimension
HIST_BINS = 5             # RGB histogram bins per channel (5^3 = 125 total)
COLOR_LOSS_WEIGHT = 0.5   # weight of the color/pattern hinge loss in training
COLOR_SCORE_WEIGHT = 0.3  # default blend of color similarity at inference
MARGIN = 0.2              # hinge-loss margin
BATCH_SIZE = 96           # bigger batch = more in-batch negatives per step
EPOCHS = 8
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
VAL_FRACTION = 0.15
IMAGE_SIZE = 224

#complete the look
OUTFIT_SLOTS = [
    ("Top",        ("Shirts & Tops",),             "core"),
    ("Bottom",     ("Pants", "Skirts", "Shorts"),  "core"),
    ("Shoes",      ("Shoes",),                     "core"),
    ("Outerwear",  ("Coats & Jackets",),           "standard"),
    ("Bag",        ("Handbags, Wallets & Cases",), "standard"),
    ("Sunglasses", ("Sunglasses",),                "full"),
    ("Necklace",   ("Necklaces",),                 "full"),
    ("Earrings",   ("Earrings",),                  "full"),
]
OUTFIT_TIERS = ["core", "standard", "full"]   # each tier includes the previous
DEFAULT_OUTFIT_TIER = "standard"

#category classificator
PROBE_VAL_FRACTION = 0.2
PROBE_EPOCHS = 300
PROBE_LR = 1e-3
PROBE_BATCH = 512
PROBE_MIN_BBOX_AREA = 0.01   
PROBE_BBOX_PAD = 0.08      



#0.6 will keep 83% of coverage
PROBE_MIN_CONFIDENCE = 0.60

# how much an item's score depends on working with the rest of the outfit
OUTFIT_COHESION_WEIGHT = 0.35
