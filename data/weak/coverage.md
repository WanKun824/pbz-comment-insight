# 弱监督覆盖率报告 v2

样本 9538 条（去重后）

## 各维度覆盖率与标签分布

| 维度 | 覆盖率 | 标注数 | 分布 |
|---|---:|---:|---|
| `valid` | 25.7% | 2454 | substantive 1355 · fragment 637 · banter 453 · spam 9 |
| `topic_l1` | 26.4% | 2520 | combat 814 · content 563 · industry 422 · tech 210 · creator 170 · business 154 · player 94 · community 93 |
| `sentiment` | 12.6% | 1203 | pos 711 · neg 284 · very_pos 163 · very_neg 45 |
| `target` | 28.0% | 2666 | game 1123 · other_game 807 · creator 364 · dev 138 · other_player 122 · none 112 |
| `speech_act` | 27.6% | 2630 | question 1012 · joke 639 · suggest 496 · report 237 · review 131 · argue 85 · share 30 |
| `is_mixed` | 81.9% | 7808 | B 7729 · A 79 |
| `has_issue` | 52.3% | 4984 | B 4556 · A 428 |
| `purchase_intent` | 43.5% | 4145 | B 3931 · A 214 |
| `churn_signal` | 42.7% | 4077 | B 3978 · A 99 |
| `is_spoiler` | 36.1% | 3447 | B 3329 · A 118 |
| `is_toxic` | 63.1% | 6014 | B 6003 · A 11 |
| `mentions_other_game` | 78.2% | 7461 | B 6080 · A 1381 |

未做规则（留人工）：is_sarcasm, is_actionable

## 每条被标注的维度数

| 维度数 | 条数 |
|---:|---:|
| 1 | 959 |
| 2 | 1425 |
| 3 | 1150 |
| 4 | 925 |
| 5 | 900 |
| 6 | 793 |
| 7 | 876 |
| 8 | 875 |
| 9 | 704 |
| 10 | 591 |
| 11 | 266 |
| 12 | 74 |

**总弱标签决策数：49,409**
（≥6 维度被标注：4179 条）

## 命中规则统计


**valid**

| 规则 | 次数 |
|---|---:|
| ≥60字有主题 | 1060 |
| ≤4字无主题 | 525 |
| 短+梗词 | 453 |
| ≥30字有主题有评价 | 295 |
| 纯表情 | 112 |
| spam词 | 9 |

**topic_l1**

| 规则 | 次数 |
|---|---:|
| combat独占x1 | 588 |
| content独占x1 | 399 |
| industry独占x1 | 328 |
| creator独占x1 | 152 |
| tech独占x1 | 152 |
| business独占x1 | 141 |
| combat独占x2 | 137 |
| content独占x2 | 98 |
| player独占x1 | 94 |
| community独占x1 | 83 |
| industry独占x2 | 73 |
| tech独占x2 | 37 |
| combat独占x3 | 29 |
| combat压倒x3vs1 | 26 |
| content压倒x3vs1 | 22 |
| content独占x3 | 19 |
| creator独占x2 | 18 |
| industry独占x3 | 16 |
| tech独占x3 | 14 |
| combat独占x4 | 11 |
| community独占x2 | 10 |
| business独占x2 | 8 |
| combat压倒x4vs1 | 7 |
| content独占x4 | 6 |
| combat独占x5 | 6 |
| content压倒x4vs1 | 6 |
| content独占x5 | 6 |
| combat压倒x5vs1 | 4 |
| business压倒x3vs1 | 3 |
| tech独占x4 | 3 |
| industry压倒x3vs1 | 3 |
| content压倒x5vs1 | 3 |
| tech压倒x3vs1 | 2 |
| business独占x3 | 2 |
| combat独占x6 | 2 |
| content压倒x6vs2 | 2 |
| industry独占x4 | 1 |
| industry压倒x4vs1 | 1 |
| combat压倒x6vs2 | 1 |
| tech压倒x6vs1 | 1 |
| content压倒x6vs1 | 1 |
| combat压倒x6vs1 | 1 |
| combat独占x7 | 1 |
| content独占x6 | 1 |
| tech独占x5 | 1 |
| combat压倒x7vs1 | 1 |

**sentiment**

| 规则 | 次数 |
|---|---:|
| 正面x1 | 414 |
| 负面x1 | 184 |
| 强正x1 | 154 |
| 正面x1+转折 | 154 |
| 负面x1+转折 | 73 |
| 正面x2 | 54 |
| 强负x1 | 44 |
| 强正x1+转折 | 42 |
| 正面x2+转折 | 27 |
| 强负x1+转折 | 14 |
| 正面x3+转折 | 8 |
| 负面x2 | 7 |
| 强正x2 | 7 |
| 负面x2+转折 | 5 |
| 强正x2+转折 | 5 |
| 正面x3 | 4 |
| 强负x2 | 1 |
| 强正x3+转折 | 1 |
| 负面x4+转折 | 1 |
| 正面x5+转折 | 1 |
| 强正x3 | 1 |
| 强正x4 | 1 |
| 正面x5 | 1 |

**target**

| 规则 | 次数 |
|---|---:|
| 游戏词独占 | 1123 |
| 仅提其他游戏 | 807 |
| UP主词独占 | 364 |
| 开发者词独占 | 138 |
| 回复+第二人称开头 | 122 |
| 纯表情 | 112 |

**speech_act**

| 规则 | 次数 |
|---|---:|
| 疑问句 | 1012 |
| 短+梗 | 639 |
| 建议词 | 496 |
| 技术词陈述 | 237 |
| 长评+评价词 | 131 |
| 回复+反驳词 | 85 |
| 第一人称经历 | 30 |