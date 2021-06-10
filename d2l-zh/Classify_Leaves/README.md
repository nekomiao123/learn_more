# 模型参数对比

| num  | name                               | batch size | imagenet top1 | Parameters   | **Leaves** acc | Public Score |
| :--: | :--------------------------------- | :--------- | :------------ | ------------ | -------------- | ------------ |
|  1   | seresnext50_32x4d                  | 32         | 81.27%        | 28 Million   | 97.31%         | 0.97204      |
|  2   | seresnext50_32x4d                  | 64         | 81.27%        | 28 Million   | 97.41%         | 0.97227      |
|  3   | efficientnetv2_rw_m                | 32         | 84.8 %        | 53 Million   | 97.22%         | 0.97000      |
|  4   | efficientnet_b3a                   | 16         | 82.25%        | 12 Million   | 97.34%         | 0.97113      |
|  5   | tf_efficientnet_b5_ns              | 16         | 86.08%        | 30 Million   | 97.20%         | 0.97000      |
|  6   | 2__mixup                           | 64         |               | 28 Millon    | 97.28%         | 0.97113      |
|  7   | 5__mixup                           | 24         |               | 30 Millon    | 97.11%         | 0.96795      |
|  8   | 4_mixup                            | 64         |               | 12 Million   | 97.23%         |              |
|  9   | AdamW_tf_efficientnetv2_s_in21ft1k | 64         | 84.9%         | 21.5 Million | 97.47%         | 0.97659      |
|  10  | AdamW_efficientnet_b3a             | 64         | 82.25%        | 12 Million   | 97.6%          | 0.97613      |
|  11  | AdamW_seresnext50_32x4d            | 64         | 81.27%        | 28 Million   | 97.58%         | 0.97750      |



## Ensemble 

| nums  | Public Score |
| ----- | ------------ |
| 2 3 4 | 0.97613      |
| 2 4 5 | 0.97590      |
|       |              |

