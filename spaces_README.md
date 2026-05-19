---
title: Cricket ML — IPL Prediction Suite
emoji: 🏏
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: IPL match winner, score, win probability & POTM predictions
---

# 🏏 Cricket ML — IPL Prediction Suite

Live ML demo serving 5 prediction models trained on IPL 2008–2019 data.

## What's inside

| Page | Model | Key metric |
|---|---|---|
| 🏏 Match Winner | XGBoost + rolling player form | AUC 0.797 |
| 📊 Score Predictor (LightGBM) | Static over-10 snapshot | MAE 18 runs |
| 🧠 Score Predictor (GRU) | 2-layer sequence model | Over-by-over inference |
| 📈 Win Probability | LightGBM ball-by-ball | AUC 0.822 |
| 🏆 POTM Predictor | XGBoost | AUC 0.972 |
| 🔍 Player Stats | Pure pandas aggregation | 180+ players |

## Source code

GitHub: [PRANAVGAWALE-DS/Cricket](https://github.com/PRANAVGAWALE-DS/Cricket)