"""SmolVLA (in-process) x LeIsaac SO101 PickOrange - part 1/3: header + policy load."""
import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import traceback

LEROBOT_SRC = "/workspace/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)
try:
    import transformers.utils as _tu

    if not hasattr(_tu, "torch_compilable_check"):
        _tu.torch_compilable_check = lambda *a, **k: None
except Exception:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SmolVLA in-process inference on LeIsaac PickOrange.")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--policy_id", type=str, default="lerobot/smolvla_base")
parser.add_argument("--task_instruction", type=str, default="Pick an orange and put it into the plate.")
parser.add_argument("--action_horizon", type=int, default=50)
parser.add_argument("--save_frames_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import leisaac  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

print("[smolvla] loading policy ...", flush=True)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors

DEVICE = "cuda"
policy = SmolVLAPolicy.from_pretrained(args_cli.policy_id)
policy.to(DEVICE)
policy.eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    args_cli.policy_id,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
    postprocessor_overrides={"device_processor": {"device": DEVICE}},
)
print(f"[smolvla] policy OK: {args_cli.policy_id} chunk={policy.config.chunk_size}", flush=True)
print(f"[smolvla] input_features: {list(policy.config.input_features.keys())}", flush=True)
TARGET = "LeIsaac-SO101-PickOrange-v0"
print(f"\n=== BUILDING {TARGET} (headless, cameras on) ===", flush=True)
env = None
try:
    env_cfg = parse_env_cfg(TARGET, num_envs=1, device="cuda:0")
    env_cfg.use_teleop_device("so101leader")
    env = gym.make(TARGET, cfg=env_cfg)
    obs, _ = env.reset(seed=args_cli.seed)
    print("reset OK", flush=True)
    unw = env.unwrapped
    print(f"robot joints: {unw.scene['robot'].data.joint_names}", flush=True)

    if args_cli.save_frames_dir:
        os.makedirs(args_cli.save_frames_dir, exist_ok=True)


    def to_chw(u8):
        t = torch.as_tensor(np.asarray(u8[0]), dtype=torch.float32, device=DEVICE) / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).contiguous()


    global_step = 0
    for ep in range(args_cli.num_episodes):
        if ep > 0:
            obs, _ = env.reset(seed=args_cli.seed + ep)
            policy.reset()
            print(f"[smolvla] episode {ep + 1} reset", flush=True)
        print(f"[smolvla] EPISODE {ep + 1}/{args_cli.num_episodes}", flush=True)
        t0 = time.time()
        finished = False
        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                finished = True
                break
            pol = obs["policy"]
            joint_rad = pol["joint_pos"]
            if not torch.is_tensor(joint_rad):
                joint_rad = torch.as_tensor(joint_rad)
            state_lerobot = torch.as_tensor(
                convert_leisaac_action_to_lerobot(joint_rad.detach()),
                dtype=torch.float32, device=DEVICE,
            )
            front = to_chw(pol["front"].detach().cpu().numpy() if torch.is_tensor(pol["front"]) else pol["front"])
            wrist = to_chw(pol["wrist"].detach().cpu().numpy() if torch.is_tensor(pol["wrist"]) else pol["wrist"])
            batch = {
                "observation.state": state_lerobot,
                "observation.images.camera1": front,
                "observation.images.camera2": wrist,
                "observation.images.camera3": torch.zeros_like(front),
                "task": [args_cli.task_instruction],
            }
            batch = preprocess(batch)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
            chunk = postprocess(chunk)[0].detach().cpu().numpy()

            horizon = min(args_cli.action_horizon, chunk.shape[0])
            for h in range(horizon):
                a_leisaac = torch.as_tensor(
                    convert_lerobot_action_to_leisaac(chunk[h][None, :]),
                    dtype=torch.float32, device=unw.device,
                )
                obs, rew, term, trunc, info = env.step(a_leisaac)
                global_step += 1
                if args_cli.save_frames_dir and global_step % 30 == 0:
                    from PIL import Image
                    p = obs["policy"]
                    for key in ("front", "wrist"):
                        im = p[key][0].detach().cpu().numpy().astype(np.uint8)
                        Image.fromarray(im).save(
                            os.path.join(args_cli.save_frames_dir, f"step{global_step:05d}_{key}.png"))
                if bool(term[0]) or bool(trunc[0]):
                    print(f"[smolvla] env ended episode at step {step}", flush=True)
                    finished = True
                    break
            if (step + 1) % 5 == 0 or finished:
                print(f"[smolvla] ep={ep + 1} policy_step={step + 1} elapsed={time.time() - t0:.0f}s",
                      flush=True)
            if finished:
                break
        print(f"[smolvla] episode {ep + 1} finished", flush=True)
    print("\nSMOLVLA LOOP DONE", flush=True)
except Exception:
    traceback.print_exc()
    print("\nSMOLVLA LOOP FAILED (see traceback above)", flush=True)
finally:
    if env is not None:
        env.close()

print("\n=== RESULT DONE ===", flush=True)
simulation_app.close()
