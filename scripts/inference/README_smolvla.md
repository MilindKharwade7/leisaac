# SmolVLA x LeIsaac PickOrange (in-process inference)

One script, no servers, no ports. Runs HuggingFace SmolVLA **inside** the Isaac
sim process (Kit main thread) on `LeIsaac-SO101-PickOrange-v0`.

## Run

```bash
cd /workspace
./isaaclab/isaaclab.sh -p leisaac/scripts/inference/run_smolvla_pick_orange.py --enable_cameras
```

Useful flags:

```bash
./isaaclab/isaaclab.sh -p leisaac/scripts/inference/run_smolvla_pick_orange.py --enable_cameras \
  --num_episodes 2 --max_steps 600 --seed 7 \
  --policy_id lerobot/smolvla_base \
  --task_instruction "Pick an orange and put it into the plate." \
  --action_horizon 50 \
  --save_frames_dir /tmp/smolvla_frames
```

Watch it live over WebRTC at the same time:

```bash
PUBLIC_IP=<host-ip> ./isaaclab/isaaclab.sh -p leisaac/scripts/inference/run_smolvla_pick_orange.py \
  --enable_cameras --livestream 1
# client -> <host-ip> (TCP 49100 + UDP 47998 forwarded)
```

## Mapping (sim <-> policy)

| sim obs (`obs["policy"]`) | policy batch key | note |
|---|---|---|
| `joint_pos` (rad, LeIsaac) | `observation.state` | via `convert_leisaac_action_to_lerobot` (rad -> motor units) |
| `front` uint8 (H,W,C) | `observation.images.camera1` | `/255`, CHW, batch dim |
| `wrist` uint8 (H,W,C) | `observation.images.camera2` | same |
| — (black frame) | `observation.images.camera3` | `smolvla_base` expects 3 cams; blank is the standard stand-in |
| `--task_instruction` | `task` | e.g. "Pick an orange and put it into the plate." |

Policy returns a 50-step chunk (`predict_action_chunk`); each action (lerobot
motor units) goes back through `convert_lerobot_action_to_leisaac` (motor units
-> rad) into `env.step`. `policy.reset()` is called on every env reset to clear
the action queue.

## Honest expectations

`lerobot/smolvla_base` was NOT trained on this kitchen/orange scene, so grasps
are exploratory. The value here: the loop, conversions, chunking and resets are
all proven — swap `--policy_id` for a fine-tuned SO101 checkpoint and the same
script does real task success.

## Container notes

* Kit python needs `PYTHONPATH=/workspace/lerobot/src` — the script adds it.
* `transformers.utils.torch_compilable_check` shim included (container has
  transformers 4.57.6, cloned lerobot expects newer).
* ~2GB VRAM for the policy + sim on the RTX 3090. Stop other Kit procs first.
