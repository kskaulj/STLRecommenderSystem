# Fashion RecommenderS ystem AKA My Master thesis.


A deep learning recommendation system built on Pinterest **Shop the Look** dataset, following the approach of *Complete the Look: Scene-based Complementary Product Recommendation* (Kang et al., CVPR 2019,
[arXiv:1812.01748](https://arxiv.org/abs/1812.01748)).

Drop an image into the web UI and the system ranks the product catalog by **style compatibility** (not visual similarity) using embeddings learned with a triplet hinge loss.

## Files

| file | purpose |
|---|---|
| `config.py` | paths + hyperparameters + outfit slot definitions |
| `download_images.py` | fetch image subset from Pinterest CDN |
| `dataset.py` | pair/scene/product datasets, bbox masking, transforms |
| `model.py` | CompatibilityNet + in-batch hinge loss |
| `train.py` | training loop, AUC evaluation, checkpointing |
| `embed_catalog.py` | precompute catalog embeddings |
| `train_category.py` | linear probe on the frozen trunk: recognises a dropped item's category |
| `metrics.py` | ranking + beyond-accuracy metrics |
| `baselines.py` | random / popularity / frozen-backbone baselines |
| `evaluate.py` | full-catalog evaluation of all methods, JSON + chart output |
| `app.py` + `templates/index.html` | drag-and-drop Flask UI: outfit composer + ranked list |

## Usage 

In PowerShell: 

```
<venv>\Scripts\Activate.ps1
python download_images.py --pairs 8000   # cca 3 min, needs internet
python train.py                          # cca 15 min on RTX 2070
python embed_catalog.py
python train_category.py                 # optional: item recognition probe
python evaluate.py --output results.json --plot recall.png
python app.py                            # open http://127.0.0.1:5000
```

In the UI, choose whether your image is an **outfit/person photo** (scene head) or a **single item** (product head), and drop the image. Two result modes:

* **Complete outfit** (default) — one product per slot, forming a wearable
  look. Pick how complete it should be (*core* = top/bottom/shoes, *standard*
  adds outerwear and a bag, *full look* adds sunglasses and jewellery) and how
  strongly the pieces should agree with each other. The result reports an
  outfit score split into "match with your image" and "pieces agree with each
  other", and marks the anchor item the rest of the look was built around.
  When the query is a single item, the category probe recognises what it is and
  leaves that slot to your own piece; you can override the detected slot from
  the dropdown.
  
* **Ranked list** — the original behaviour: top-k most compatible products,
  optionally restricted to one category.
