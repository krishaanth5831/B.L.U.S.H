# Labelling guide

Keep this open while labelling. Label one session in one sitting.

## Stages → model classes

Label every fruit with one of the six standard tomato colour stages in
`data/stages.csv`. The YOLO `.txt` gets the 3-class id.

| Colour stage | Looks like | Class id | Model class |
|---|---|---|---|
| green | Fully green | 0 | `unripe` |
| breaker | First blush of colour at the bottom | 0 | `unripe` |
| turning | 10–30 % coloured | 1 | `turning` |
| pink | 30–60 % coloured | 1 | `turning` |
| light_red | 60–90 % red | 2 | `ripe` |
| red | 90 %+ red | 2 | `ripe` |

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
