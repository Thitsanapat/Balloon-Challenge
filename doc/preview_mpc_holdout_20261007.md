# Preview MPC + frozen PPO: ten-seed holdout

Ten previously unseen Scenario 4 seeds were evaluated in paired runs on the
teacher's machine with one CPU worker and no GPU. The raw metrics and rollup
are preserved in the ignored local artifact
`BalloonPoppingGymEnv/evaluation/results/preview_mpc_holdout10_complete_20261006.zip`
(SHA-256 `ee074de934d61d119e2d4aa38d851aee30e144163872ecc895a074e07b896419`).

| Seed | Parent PPO | Preview MPC + PPO |
| ---: | ---: | ---: |
| 1638895464 | 4 | 3 |
| 1584252769 | 3 | 4 |
| 1722428225 | 6 | 4 |
| 1859368188 | 6 | 3 |
| 1570091972 | 4 | 5 |
| 1824032641 | 4 | 4 |
| 1945086148 | 4 | 2 |
| 1533739991 | 2 | 2 |
| 1676901674 | 3 | 4 |
| 1830415931 | 5 | 4 |

Mean: **4.1** versus **3.5** pops; preview MPC wins 3, ties 2, loses 5.
The earlier five-seed validation had suggested an improvement, but this
independent holdout does not confirm it. Do **not** promote this candidate to
the qualification launcher. Neither result guarantees a live-round score.
