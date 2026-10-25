# 7B v2 run results

- teacher past: past-pick 1.000, lean +15.39 (240 judgments)
- teacher present: past-pick 0.000, lean -17.00 (240 judgments)

| student | epoch | past-pick | lean |
|---|---|---|---|
| past | 1 | 0.621 | +0.974 |
| past | 2 | 0.642 | +1.275 |
| past | 3 | 0.667 | +1.422 |
| past | 4 | 0.696 | +1.544 |
| past | 5 | 0.700 | +1.617 |
| past | 6 | 0.696 | +1.718 |
| past | 7 | 0.704 | +1.747 |
| past | 8 | 0.704 | +1.833 |
| present | 1 | 0.512 | +0.142 |
| present | 2 | 0.492 | +0.038 |
| present | 3 | 0.504 | -0.058 |
| present | 4 | 0.483 | -0.118 |
| present | 5 | 0.475 | -0.185 |
| present | 6 | 0.467 | -0.233 |
| present | 7 | 0.463 | -0.249 |
| present | 8 | 0.446 | -0.296 |
| base | 1 | 0.562 | +0.893 |
| base | 2 | 0.562 | +0.908 |
| base | 3 | 0.562 | +0.871 |
| base | 4 | 0.558 | +0.844 |
| base | 5 | 0.562 | +0.889 |
| base | 6 | 0.571 | +0.864 |
| base | 7 | 0.567 | +0.897 |
| base | 8 | 0.567 | +0.906 |

Final epoch comparisons (240 judgments, 95% intervals over 120 pairs):

| comparison | past-pick (points) | lean |
|---|---|---|
| past - base | +13.75 [+8.75, +18.75] | +0.93 [+0.55, +1.29] |
| present - base | -12.08 [-17.08, -7.08] | -1.20 [-1.69, -0.74] |
| past - present | +25.83 [+20.42, +31.25] | +2.13 [+1.94, +2.33] |
