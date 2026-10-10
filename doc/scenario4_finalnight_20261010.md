# Final-night Scenario 4 freeze — 10 October 2026

## Selection evidence already available

The pure joint refresh preserves the currently observed route and interception
deadlines, but re-solves shared arrival velocity/acceleration when the committed
leg becomes infeasible. It retains the existing navigation, wind estimation and
guarded terminal control. It is not a new DRL policy and requires no GPU.

Four predeclared fresh paired fields (`pure_joint_holdout4_recovered_20261009`):

| Seed | Previous guarded | Pure joint |
| ---: | ---: | ---: |
| 605131410 | 4 | 4 |
| 435857031 | 1 | 5 |
| 927297526 | 4 | 6 |
| 298628555 | 3 | 4 |

Total **12 vs 19**, mean **3 vs 4.75**; three wins and one tie. This is a small
sample, not 19 pops in one flight or a guarantee for the qualification seed.
Development seeds 0 and 2 scored 6 and 4 with the same modular policy.

The reproducible standalone v2 matches the full seed-2 modular flight, including
telemetry. Its SHA256 is
`9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9`.
`submission_scenario4_pure_joint_v2.py` is byte-identical to that reviewed artifact.
Do not use the earlier candidate v1: it failed planner-hook/flight parity.

## Final bounded experiment (predeclared)

`Scenario4ProactiveJointAgent` attempts the same joint solve *before* ordinary
refresh becomes infeasible, if the observed endpoint moves by at least half
the balloon radius and interception is 1–6 seconds away. A failed solve restores
the already-feasible ordinary refresh. It does not change target order or
deadlines and does not access seeds, hidden state, or recorded trajectories.

Run `python -m scripts.validate_proactive_joint`. Two CPU workers; no GPU.
Existing development seeds 0 and 2 only. Baseline references are complete
official-environment flights, with dependency hashes checked before reuse.
Candidate flights are fresh official simulations, not trajectory-cache reuse.
Gate: total gain >=2, no seed loses more than one pop, and at least one proactive
repair actually accepted. A pass requires a separate fresh paired validation
before promotion. A failure leaves the verified pure joint candidate intact.

Navigation/filter and angular-response adaptation from 9 October did not beat
the existing joint policy and are not part of either candidate.

## Readiness and limitations

Seventeen targeted unit tests passed for proactive fallback, joint repair,
standalone assembly and live-path configuration, seed registry and
qualification-run failure handling.
The live-path candidate passed official core/installed RocketPy preflight.
This does **not** establish that every installed dependency matches `uv.lock`;
keep the tested environment unchanged rather than upgrade on event day.

The existing registry has six coverage warnings about older unlogged PPO seeds.
Do not claim a complete historical seed inventory. Every new test must be
recorded; the standard launcher refuses known used/reserved seeds.

`scripts.validate_joint_live_ready` performs a local official-packaging rehearsal
on a fresh development seed with explicit `--confirm-local-packaging`. Its
runner mode is `live` only to exercise JSON packaging: it never uploads and is
not an organizer-assigned evaluation. The resulting submission contains the
team secret; do not show its contents or publish it.

## Final experiment outcome

| Development seed | Pure joint reference | Proactive joint |
| ---: | ---: | ---: |
| 0 | 6 | 5 |
| 2 | 4 | 2 |

The proactive screen **failed**, total 10 -> 7. Both flights exercised one
accepted proactive repair, so this is not an unexercised feature. Other requests
fell back to the feasible ordinary refresh (10 and 7 times). Saturated-step
counts increased from 674 to 1013 and 1177 to 1457 respectively. These are
flight-level diagnostics, not proof of a single causal mechanism. The scheme
did not improve capture and is excluded from the competition configuration.
Full reports: `evaluation/results/proactive_joint_dev_20261010/summary.json`.

The selected candidate remains **pure joint refresh v2**, not proactive repair,
adaptive angular gain, or the sensor-filter experiments. There is no new claim
of 20 pops. Its paired-field evidence is the four-flight table above.

## Reproducing the selected source

Use the official v0.2.2 repository with its pinned ActiveRocketPy submodule.
Extract the final source archive over that checkout. The standalone agent
depends on the official `BaseAgent`, NumPy and SciPy, not local result files,
seed logs or trained weights. The manifest records installed package versions;
it is not an assertion that those versions equal the official lockfile.

The included builder can recreate the standalone at a new path:

```text
python -m scripts.build_scenario4_joint_submission_agent --output review/rebuilt_joint.py
```

Check the SHA256 against the value above. Do not overwrite a tested artifact.
For offline development evaluation, the diagnostic runner takes an explicit
seed; for the actual event use only the qualification launcher, which checks
seed reuse and creates the official submission object.

## Operator reminders (Thai)

- วันจริงใช้ seed ที่ผู้จัดให้เท่านั้น ไม่ใช้ JSON จากการซ้อมคืนนี้แทน
- เปิด `run_qualification.cmd` ใส่เฉพาะตัวเลข seed แล้วพิมพ์ `LIVE` ในช่อง mode
  และ `RUN` ในช่องยืนยัน รอจนขึ้น `QUALIFICATION RESULT` กับ `UPLOAD THIS JSON`
- ส่งเฉพาะไฟล์ที่โปรแกรมระบุหลัง `UPLOAD THIS JSON` ไม่ใช่ trajectory หรือ ZIP
  และอย่าเปิดเนื้อหา JSON ขณะแชร์จอ เพราะมีรหัสทีมอยู่ข้างใน
- `run_qualification_guarded_backup.cmd` เรียกตัว guarded เก่าไว้เป็นสำรอง
  ให้ตัดสินใจเลือกตัวก่อนเริ่มรัน ไม่สลับตัวแล้วลองซ้ำบน seed วันจริงเอง
- ถ้ารันล้มเหลว ให้เก็บข้อความผิดพลาดและคุยกับผู้จัดก่อน เรื่องรันซ้ำ/จับ seed ใหม่
  หากมีการแก้ seed ค้างจากการหยุดกลางคัน ใช้ `run_qualification.cmd --recover`
- เก็บ `.venv` เดิมไว้ ไม่อัปเดตแพ็กเกจหรือแก้ฟิสิกส์ของ simulator ก่อนแข่ง

## Final selection and packaging rehearsal

The standard launcher now defaults to
`submission_scenario4_pure_joint_launch42.yaml`. The old guarded module and
configuration are unchanged; the separate backup CMD selects them explicitly.

Local official-runner rehearsal seed **741093826** scored **4 pops** and created
a validated submission object. The scenario file was restored byte-for-byte;
no active recovery marker remains. This seed is now used. The clean receipt is
`qualification/finalnight_joint_readiness.json`. The JSON SHA256 is
`555f637980c5a6c5e7726016e171858b965545bd6b7402054199d144d45b24e4`.
The receipt gives the JSON path without exposing credentials. Nothing was uploaded.

The packer's optional online integrity lookup was blocked by the local network.
The launcher's local v0.2.2 core-file and installed RocketPy checks passed.
This warning did not prevent packaging and is not a dependency-lockfile check.

Final source archive: `review/scenario4_finalnight_joint_source_20261010.zip`.
It contains explicit code/config/docs, modular dependencies and a runtime/hash
manifest, but no packed submission JSON or credential file. This archive is
for source review; it is not the evaluation upload. Both archive CRC and each
listed file hash are checked by `scripts.package_finalnight_joint`.
