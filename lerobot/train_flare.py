import os
import subprocess

# Create log directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# train params
policy_type = "flare"
steps = 200000
eval_freq = 10000
visualize_freq = 2500
save_freq = 100000 # TODO
batch_size = 64
optimizer_lr = 1e-4
optimizer_lr_backbone = 1e-4
use_ema = True
grad_clip_norm = 1.0

# wandb params
wandb_enable = True
project = "cv_final_project"

# dataset/env params
dataset_n_episodes = 100
env_type = "av_aloha"
dataset_repo_id = "iantc104/av_aloha_sim_thread_needle_240x320"
use_lerobot_dataset=False
env_task = "thread-needle-v1"
env_episode_length = 11
fps = 8.3333333
eval_n_episodes = 50
eval_batch_size = 10

# policy params
drop_n_last_frames = 24
n_obs_steps = 1
n_obs_step_size = 1
# vision
# resize_shape = [int(48/.9), int(64/.9)]
# crop_shape = [int(48), int(64)]
resize_shape = [240, 320]
crop_shape = [228, 304]
crop_is_random = True
if not crop_is_random and crop_shape != resize_shape:
    raise ValueError("If crop_is_random is False, crop_shape probably should equal to resize_shape")
use_spatial_softmax = False
out_dim = 512
# net
flow_net = "transformer"
n_decoder_layers = 6
# gaze params
use_gaze = True
gaze_crop_shape = [48, 64]
gaze_add_small_features = False

job_name = f"adaln_1obs1step+fullimage"

start = 0
end = 0
gpu = 0 # -1 for sequential 
seed = 0 # -1 for sequential 
session_index = 0 # -1 for sequential 
# Loop through values of n
for n in range(start, end+1):
    gpu_index = gpu if gpu >= 0 else n
    s = str(seed if seed >= 0 else n)
    sess_index = session_index if session_index >= 0 else n

    command = (
        f"MUJOCO_EGL_DEVICE_ID={gpu_index} CUDA_VISIBLE_DEVICES={gpu_index} "
        f"python lerobot/scripts/train.py "
        f"--wandb.enable={str(wandb_enable).lower()} "
        f"--wandb.project={project} "
        f"--policy.type={policy_type} "
        f"--steps={steps} "
        f"--eval_freq={eval_freq} "
        f"--save_freq={save_freq} "
        f"--visualize_freq={visualize_freq} "
        f"--batch_size={batch_size} "
        f"--policy.optimizer_lr={optimizer_lr} "
        f"--policy.optimizer_lr_backbone={optimizer_lr_backbone} "
        f"--policy.use_ema={str(use_ema).lower()} "
        f"--policy.grad_clip_norm={grad_clip_norm} "
        f"--seed={s} "
        f"--dataset.repo_id={dataset_repo_id} "
        f"--dataset.use_lerobot_dataset={str(use_lerobot_dataset).lower()} "
        f"--dataset.n_episodes={dataset_n_episodes} "
        f"--env.type={env_type} "
        f"--env.task={env_task} "
        f"--env.episode_length_s={env_episode_length} "
        f"--eval.n_episodes={eval_n_episodes} "
        f"--eval.batch_size={eval_batch_size} "
        f"--env.fps={fps} "
        f"--policy.fps={fps} "
        f"--policy.drop_n_last_frames={drop_n_last_frames} "
        f"--policy.n_obs_steps={n_obs_steps} "
        f"--policy.n_obs_step_size={n_obs_step_size} "
        f"--policy.vision.resize_shape '{str(resize_shape)}' "
        f"--policy.vision.crop_shape '{str(crop_shape)}' "
        f"--policy.vision.crop_is_random={str(crop_is_random).lower()} "
        f"--policy.vision.use_spatial_softmax={str(use_spatial_softmax).lower()} "
        f"--policy.vision.out_dim={out_dim} "
        f"--policy.vision.use_gaze={str(use_gaze).lower()} "
        f"--policy.vision.gaze_crop_shape '{str(gaze_crop_shape)}' "
        f"--policy.vision.gaze_add_small_features={str(gaze_add_small_features).lower()} "
        f"--policy.flow_net={flow_net} "
        f"--policy.n_decoder_layers={n_decoder_layers} "
        f"--job_name={job_name}_seed{s} "
    )

    session_name = f"session_{sess_index}"
    log_file = f"logs/session_{sess_index}.log"
    
    tmux_command = (
        f'tmux new-session -d -s {session_name} '
        f'bash -c "{command} > {log_file} 2>&1; '
        f'if [ $? -ne 0 ]; then echo \'Command failed\' >> {log_file}; fi"'
    )

    subprocess.run(tmux_command, shell=True, check=True)
    print(f"Started tmux session '{session_name}' with command: {command}")
