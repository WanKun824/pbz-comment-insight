# 训练集统计

封顶 cap=450　seed=0　test=12% dev=10%

## 切分规模

| split | 条数 | 决策数 |
|---|---:|---:|
| train | 5002 | 9689 |
| dev | 633 | 1188 |
| test | 771 | 1485 |

## 封顶前后（全量）

| 维度 | 标签 | 原始 | 保留 |
|---|---|---:|---:|
| `valid` | banter | 453 | 450 ⬇ |
| `valid` | fragment | 637 | 450 ⬇ |
| `valid` | spam | 9 | 9 |
| `valid` | substantive | 1355 | 450 ⬇ |
| `topic_l1` | 商业产品 | 154 | 154 |
| `topic_l1` | 战斗 | 814 | 450 ⬇ |
| `topic_l1` | 社区舆论 | 93 | 93 |
| `topic_l1` | 内容品质 | 563 | 450 ⬇ |
| `topic_l1` | 视频与UP主 | 170 | 170 |
| `topic_l1` | 行业生态 | 422 | 422 |
| `topic_l1` | 玩家自述 | 94 | 94 |
| `topic_l1` | 技术性能 | 210 | 210 |
| `sentiment` | neg | 284 | 284 |
| `sentiment` | pos | 711 | 450 ⬇ |
| `sentiment` | very_neg | 45 | 45 |
| `sentiment` | very_pos | 163 | 163 |
| `target` | creator | 364 | 364 |
| `target` | dev | 138 | 138 |
| `target` | game | 1123 | 450 ⬇ |
| `target` | none | 112 | 112 |
| `target` | other_game | 807 | 450 ⬇ |
| `target` | other_player | 122 | 122 |
| `speech_act` | argue | 85 | 85 |
| `speech_act` | joke | 639 | 450 ⬇ |
| `speech_act` | question | 1012 | 450 ⬇ |
| `speech_act` | report | 237 | 237 |
| `speech_act` | review | 131 | 131 |
| `speech_act` | share | 30 | 30 |
| `speech_act` | suggest | 496 | 450 ⬇ |
| `is_mixed` | A | 79 | 79 |
| `is_mixed` | B | 7729 | 450 ⬇ |
| `has_issue` | A | 428 | 428 |
| `has_issue` | B | 4556 | 450 ⬇ |
| `purchase_intent` | A | 214 | 214 |
| `purchase_intent` | B | 3931 | 450 ⬇ |
| `churn_signal` | A | 99 | 99 |
| `churn_signal` | B | 3978 | 450 ⬇ |
| `is_spoiler` | A | 118 | 118 |
| `is_spoiler` | B | 3329 | 450 ⬇ |
| `is_toxic` | A | 11 | 11 |
| `is_toxic` | B | 6003 | 450 ⬇ |
| `mentions_other_game` | A | 1381 | 450 ⬇ |
| `mentions_other_game` | B | 6080 | 450 ⬇ |

## 各 split 标签分布


**valid**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| banter | 342 | 53 | 55 |
| fragment | 331 | 54 | 65 |
| spam | 6 | 2 | 1 |
| substantive | 359 | 42 | 49 |

**topic_l1**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| 商业产品 | 127 | 13 | 14 |
| 战斗 | 359 | 43 | 48 |
| 社区舆论 | 69 | 11 | 13 |
| 内容品质 | 346 | 56 | 48 |
| 视频与UP主 | 130 | 19 | 21 |
| 行业生态 | 333 | 38 | 51 |
| 玩家自述 | 76 | 6 | 12 |
| 技术性能 | 183 | 10 | 17 |

**sentiment**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| neg | 218 | 25 | 41 |
| pos | 346 | 50 | 54 |
| very_neg | 38 | 4 | 3 |
| very_pos | 124 | 17 | 22 |

**target**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| creator | 282 | 36 | 46 |
| dev | 99 | 22 | 17 |
| game | 366 | 37 | 47 |
| none | 85 | 13 | 14 |
| other_game | 348 | 44 | 58 |
| other_player | 96 | 12 | 14 |

**speech_act**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| argue | 73 | 7 | 5 |
| joke | 347 | 45 | 58 |
| question | 335 | 53 | 62 |
| report | 193 | 13 | 31 |
| review | 97 | 13 | 21 |
| share | 22 | 3 | 5 |
| suggest | 367 | 33 | 50 |

**is_mixed**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 57 | 6 | 16 |
| B | 354 | 42 | 54 |

**has_issue**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 344 | 30 | 54 |
| B | 342 | 47 | 61 |

**purchase_intent**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 162 | 23 | 29 |
| B | 361 | 31 | 58 |

**churn_signal**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 75 | 18 | 6 |
| B | 355 | 43 | 52 |

**is_spoiler**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 83 | 14 | 21 |
| B | 358 | 41 | 51 |

**is_toxic**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 9 | 1 | 1 |
| B | 363 | 40 | 47 |

**mentions_other_game**

| 标签 | train | dev | test |
|---|---:|---:|---:|
| A | 359 | 39 | 52 |
| B | 370 | 39 | 41 |