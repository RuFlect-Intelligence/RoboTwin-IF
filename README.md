# RoboTwin-IF Mixed 50

Mixed 50 evaluates whether a robot policy grounds language in a shared initial scene. The frozen panel has 25 Pick scenes and 25 Placement scenes. Each scene has two original instructions and ten seeds (0–9), for 1,000 episodes per policy. The two instructions for a scene and seed share the same initial state and camera observations.

## Get the scene and asset bundle

Download `robotwin-if-mixed50-v1-data.tar.gz` separately from the source tree. Extract it into the checkout. The archive contains the frozen panel, the required M10 realized families, M11 units, M12 scene specifications, M13 task context, and RoboTwin object and ALOHA assets. Extraction creates `panel/`, `assets/`, and `SHA256SUMS`. The code pins the checksum list.

```bash
tar -xzf /path/to/robotwin-if-mixed50-v1-data.tar.gz
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m robotwin_if validate
```

`validate --bundle /path/to/extracted/bundle` can also check a separate data directory on a CPU machine. For simulation, extract into the checkout or link its `assets/` to the selected bundle's assets directory.

## Run a policy

Run on Linux with a CUDA-capable NVIDIA GPU, a working Vulkan driver, and a CUDA toolkit compatible with PyTorch and CuRobo. Install the simulator dependencies below in the same Python 3.11 environment. For a different CUDA toolkit choose the matching PyTorch wheel.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-sim.txt
git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git /tmp/robotwin-if-curobo
python -m pip install --no-build-isolation /tmp/robotwin-if-curobo
```

The hold-position example exercises the policy interface and the simulator. It holds the initial robot pose for both instructions.

```bash
python -m robotwin_if run --policy examples.hold_position --smoke --output runs/smoke
python -m robotwin_if report --output runs/smoke
```

Omit `--smoke` for the full 1,000 episodes. Run one policy per output directory. Repeat the same command to reuse verified episodes and retry incomplete ones. Use `--config options.json` to pass a JSON object to your policy. Supply `--policy-revision` with a version or digest for external policy code and weights so that a changed policy starts a separate run.

### Policy contract

Put your own module on `PYTHONPATH` and pass its dotted name to `--policy`. The module exports `create_policy(config)`, returning an object with `reset(seed)` and `act(observation, instruction)`. An optional `close()` releases resources when a run ends. See [examples/hold_position.py](examples/hold_position.py).

The observation contains:

- `rgb`: uint8 H×W×3 frames from `head_camera`, `left_camera`, and `right_camera`.
- `joint_position`: measured left arm joints and gripper, then right arm joints and gripper.
- `endpose`: the two current RoboTwin end-effector poses and gripper openings.

The action is a finite vector of 14 absolute joint setpoints in the same arm order. Use radians for the six joints per arm and a normalized opening in `[0, 1]` for each gripper, with 0 closed. The evaluator applies one command every 40 ms for up to 1,000 commands. A policy can buffer a predicted action chunk and return its next command on each call.

### Metrics

| Subset | Grounding | Success |
| --- | --- | --- |
| Pick | An actual robot gripper contacts the instructed object with summed contact impulse above `1e-6` at any point in the rollout. | The canonical checker verifies a stable grasp and lift by either arm, including scene integrity. |
| Placement | After verified acquisition of the instructed object, its footprint satisfies the instructed destination relation and the bottom is within 2 cm of the support plane for at least 0.5 consecutive simulated seconds. | The canonical checker verifies transfer, release, stable placement in the instructed relation, and scene integrity. |

Grounding and Success are scored independently. The Placement scorer reads the destination geometry from canonical physical checks sampled every 20 physics steps. Loss of the relation or the height condition resets its dwell timer.

`report` writes `report.json` with counts and rates for the full roster, each task type, and the two instruction groups. The report verifies paired initial-state and observation hashes. Missing episodes and execution errors are reported separately. A group's rates are emitted when all its planned episodes have valid results.

The code is derived from RoboTwin 2.0. See [LICENSE](LICENSE) for the software license. The data archive contains simulator assets and benchmark scenes. Consult the original asset sources for their terms.
