# Public datasets (local only)

Everything in this folder except this README is gitignored. Download here, one
folder per dataset, so a commercial model can be rebuilt without the
non-commercial sets.

| Folder | Dataset | Use | Class mapping | Licence |
|---|---|---|---|---|
| `laboro/` → `laboro_yolo/` | [Laboro Tomato](https://datasetninja.com/laboro-tomato) | Cherry (`l_`) classes only, drop all `b_` | `l_green → unripe`, `l_half_ripened → turning`, `l_fully_ripened → ripe` | CC BY-NC-SA 4.0, **non-commercial** |
| `mendeley/` → `mendeley_yolo/` | [Cherry Tomato Datasets](https://data.mendeley.com/datasets/ffgp73gfsr) (2024) | Five cherry varieties, greenhouse | `Green → unripe`, `Half_ripened → turning`, `Fully_ripened → ripe` | CC BY 4.0 |
| `tomatod/` | [tomatOD](https://github.com/up2metric/tomatOD) | Robot-arm viewpoint, normal-size tomatoes | `unripe`, `semi-ripe → turning`, `fully ripe → ripe` | CC BY-NC-SA 4.0, **non-commercial** |
| `rob2pheno/` | [Rob2Pheno](https://data.4tu.nl/datasets/c1f527d9-bd40-46d3-9afd-13e401f9479f) | RGB-D, Enza Zaden greenhouse | Ripeness classes unverified | CC BY 4.0 |

Pipeline per set (Laboro shown):

```bash
python ripeness/scripts/convert_coco.py ripeness/data/public/laboro/annotations ripeness/data/public/laboro_yolo
#  -> copy/move the images into laboro_yolo/images/ so they mirror laboro_yolo/labels/
python ripeness/scripts/remap_classes.py ripeness/data/public/laboro_yolo/labels laboro
```

Then list the `*_yolo` folder under `public:` in `../splits.yaml`.
