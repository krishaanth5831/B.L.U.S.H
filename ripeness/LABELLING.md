# Labelling guide

Keep this open while labelling. Label one session in one sitting.

## Stages → model classes

Label every fruit with one of the six standard tomato colour stages in
`data/stages.csv`. The YOLO `.txt` gets the class id. Anything that looks like a
tomato but isn't gets class 3 (see "Non-tomato sessions").

| Colour stage | Looks like | Class id | Model class |
|---|---|---|---|
| green | Fully green | 0 | `unripe` |
| breaker | First blush of colour at the bottom | 0 | `unripe` |
| turning | 10–30 % coloured | 1 | `turning` |
| pink | 30–60 % coloured | 1 | `turning` |
| light_red | 60–90 % red | 2 | `ripe` |
| red | 90 %+ red | 2 | `ripe` |
| not_tomato | A look-alike: ball, pepper, citrus, apple, berry, … | 3 | `not_tomato` |

The sidecar exists so the pick line can move later (e.g. stop picking `light_red`)
without relabelling anything.

## Rules

1. **Green or yellow shoulders around the stem = never `ripe`.** Tomatoes ripen
   bottom-up, so the shoulders turn last.
2. **Box partly hidden fruit** if about a third or more is visible. Skip fruit
   that's mostly behind leaves or other fruit.
3. Blotchy or streaked colour = `turning`, even if it's mostly red.
4. Box the fruit only: no calyx, no stem.
5. Unsure between two stages? Pick the **less ripe** one. That's the safe
   direction for the error that matters.

## `stages.csv`

One row per box, in the same order as the lines in that frame's YOLO `.txt`:

```
session,image,box_idx,stage
s20261012_1430,1760275800123.jpg,0,red
s20261012_1430,1760275800123.jpg,1,breaker
```

## Public-data check

Before merging a public set, look at 20 of its images and check how it labelled
green-shouldered fruit. If it calls them ripe, relabel those boxes (rule 1).
Laboro's classes are by % red (<30 / 30–89 / ≥90). That's close enough to merge,
and its "fully ripened" sits at the strict end of our `ripe`, which is the safe side.

## Non-tomato sessions (negatives)

The model has to learn that red, yellow or green alone doesn't make a tomato. Capture
whole sessions of look-alikes in front of the claw camera: red / yellow / green balls,
peppers, apples, citrus, toys. Use the same locked camera settings.

- **Box every look-alike as `not_tomato` (class 3)**, with stage `not_tomato` in
  `stages.csv`. That keeps `stages.csv` one row per box, in `.txt` line order.
- A frame with a tomato **next to** a look-alike: box both, each with its own class.
- Box only things that could pass for a tomato (round, tomato-sized, red / yellow /
  orange / green). Leaves, stems, the gripper and the table stay unboxed (background).
- A frame with nothing in it gets an **empty** `.txt` and no `stages.csv` rows.
- `not_tomato` is not a safety net: no box at all still means "not a tomato". The gate
  only ever picks `ripe`.
- Put one negatives session in `val` in `splits.yaml`, like any other session.
