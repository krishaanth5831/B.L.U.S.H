# Own claw-camera data

The data that makes the model work on the SO-101. Public sets come from other
cameras, distances and lighting.

```
own/
├── images/<session>/<ms_timestamp>.jpg   written by scripts/capture_claw.py
└── labels/<session>/<ms_timestamp>.txt   YOLO export, one line per box: class cx cy w h (normalised)
```

- `<session>` = `sYYYYMMDD_HHMM`, one per capture sitting. Sessions are the unit of
  the train/val/test split (`../splits.yaml`).
- Change **one** thing per session (vine, day, angle) and note it in the table below.
- Frames pulled from recorded picking episodes count too. Give them their own
  session folder named after the episode dataset.

## Session log

| Session | Date | Vines (picked on) | Lighting / cam settings | What changed | Frames | Labelled |
|---|---|---|---|---|---|---|
| | | | | | | |
