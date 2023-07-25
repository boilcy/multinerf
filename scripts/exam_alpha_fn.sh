#!/bin/bash

run_experiment() {
  local alpha0=$1
  local linear=$2

  python localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/home/yc/code/multinerf/datasets/real_data/'" \
    --gin_bindings "Config.checkpoint_dir = '/home/yc/code/multinerf/nerf_results/my_real_10/'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/yc/code/multinerf/nerf_results/localization_real/frequency_threshold/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.pose_lr_init = 0.01" \
    --gin_bindings "Config.pose_lr_final = 0.006" \
    --gin_bindings "Config.pose_sampling_strategy = 'random'" \
    --gin_bindings "Config.pose_delta_x = 0.5" \
    --gin_bindings "Config.pose_delta_y = 0.0" \
    --gin_bindings "Config.pose_delta_z = 0.0" \
    --gin_bindings "Config.pose_delta_phi = 0" \
    --gin_bindings "Config.pose_delta_theta = 0" \
    --gin_bindings "Config.pose_delta_psi = 0" \
    --gin_bindings "Config.pose_w_alpha = True" \
    --gin_bindings "Config.pose_manifold = True" \
    --gin_bindings "Config.pose_alpha0 = $alpha0" \
    --gin_bindings "Config.pose_alpha_linear = $linear" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count + 1))
}

# 初始化实验计数器
experiment_count=18

linen_values=("True" "False")
alpha0_values=(0.5)

for alpha0 in "${alpha0_values[@]}"; do
  for linen in "${linen_values[@]}"; do
    run_experiment $alpha0 $linen
  done
done