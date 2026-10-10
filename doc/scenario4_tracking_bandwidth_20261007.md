# Scenario 4 tracking-bandwidth experiment

The selected guarded agent scored 3 on development seed 2. A read-only
telemetry rerun reproduced 3. For two unpopped targets, the closest official
true distances were 1.95 and 1.74 m, just outside the public 1.5 m collision
radius. At the corresponding closest *observed* samples, reference tracking
errors were 4.35 and 4.98 m, while commanded throttle was about 0.80 and
0.66. Observed distance can disagree with official truth because GNSS is
noisy; the telemetry does not prove that tracking alone caused either miss.

The current controller's position/velocity feedback natural frequency is
1.3 rad/s, while its thrust allocator and attitude controller enforce public
limits. Test one predeclared candidate, `tracking_frequency=1.6`, with all
other parameters identical on already-used development seed 2. A stronger
tracking term may use unused thrust to reduce late path error, but may also
increase saturation and degrade flight. Promote to development seeds 0 and 1
only if seed 2 scores at least 4 without worsening the mean path error or
major saturation count. Any live selection would need later independent
paired-seed evidence and a new standalone/source review package. The current
qualification agent and launcher are unchanged by this experiment.

## Result and decision

The fresh official-environment telemetry flight scored 4 on development
seed 2, compared with 3 for the selected agent. However, mean reference
tracking error **rose** from 0.564 to 0.709 m, and thrust-allocation
saturation steps rose from 1,155 to 1,346. The popped IDs also changed from
59/81/88 to 59/90/23/45. It therefore failed the predeclared tracking and
saturation gates despite the one-pop score gain. A changed route or lucky
collision is possible; one seed cannot establish the cause. No additional
seeds were run, and the live agent/launcher remain unchanged. The ignored
local result is `s4_tracking_wn16_dev_seed2_20261007.json`.
