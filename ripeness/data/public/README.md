# Public datasets (local only)

Everything in this folder except this README is gitignored. Download here, one
folder per dataset, so a commercial model can be rebuilt without the
non-commercial sets.

| Folder | Dataset | Status (checked 2026-10-09) | Licence |
|---|---|---|---|
| `laboro/` → `laboro_yolo/` | [Laboro Tomato](https://github.com/laboroai/LaboroTomato), [zip](http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip) (1.6 GB) | **Used.** 804 images, 6 classes (`b_`/`l_` × green / half_ripened / fully_ripened), category ids 1–6 in that order. Cherry and regular tomatoes both mapped to our 3 classes. | CC BY-NC-SA 4.0, **non-commercial** |
| `mendeley/` | [Cherry Tomato Datasets](https://data.mendeley.com/datasets/ffgp73gfsr), `dataset1.rar` | **Not usable for ripeness.** Already YOLO, but only 2 classes: 0 = orange/red fruit, 1 = calyx. Green and yellow fruits are left **unlabelled**, so it would teach "green fruit = background". 1,500 images incl. noise-augmented copies (`_1`, `_2`). The calyx boxes might help find the picking point later. | CC BY 4.0 |
| `tomatod/` | [tomatOD](https://github.com/up2metric/tomatOD) | **Unavailable.** Both S3 download links return 403. | CC BY-NC-SA 4.0 |
| `rob2pheno/` | [Rob2Pheno](https://data.4tu.nl/datasets/c1f527d9-bd40-46d3-9afd-13e401f9479f) | Not checked. Ripeness classes unverified. | CC BY 4.0 |

Laboro:

```bash
cd ripeness/data/public/laboro && curl -LO http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip
bsdtar -xf laboro_tomato.zip && cd -           # bsdtar ships with miniforge
python ripeness/scripts/prepare_laboro.py      # -> laboro_yolo/, Laboro's test becomes our val
```

⚠️ Don't use `convert_coco.py` on Laboro: Ultralytics' converter defaults to COCO-91 → 80 id
remapping and scrambles the classes. `prepare_laboro.py` converts directly.

Then list the `*_yolo` folder under `public:` in `../splits.yaml`.
