#!/bin/bash

run_experiment() {
  local name="lr_x_1"
  local lr_init=$1
  local lr_final=$2
  local method=$3

  python localization.py --gin_configs configs/pose.gin \
    --gin_bindings "Config.data_dir = '/home/yc/code/multinerf/datasets/real_data/'" \
    --gin_bindings "Config.checkpoint_dir = '/home/yc/code/multinerf/nerf_results/my_real_10/'" \
    --gin_bindings "Config.render_path = False" \
    --gin_bindings "Config.render_dir = '/home/yc/code/multinerf/nerf_results/localization_real/$name/'" \
    --gin_bindings "Config.render_path_frames = 10" \
    --gin_bindings "Config.render_video_fps = 2" \
    --gin_bindings "Config.batch_size = 1984" \
    --gin_bindings "Config.pose_max_steps = 1000" \
    --gin_bindings "Config.pose_lr_init = $lr_init" \
    --gin_bindings "Config.pose_lr_final = $lr_final" \
    --gin_bindings "Config.pose_sampling_strategy = 'random'" \
    --gin_bindings "Config.pose_delta_x = 0.8" \
    --gin_bindings "Config.pose_delta_y = 0.0" \
    --gin_bindings "Config.pose_delta_z = 0.0" \
    --gin_bindings "Config.pose_delta_phi = 0.0" \
    --gin_bindings "Config.pose_delta_theta = 0.0" \
    --gin_bindings "Config.pose_delta_psi = 0.0" \
    --gin_bindings "Config.pose_w_alpha = True" \
    --gin_bindings "Config.pose_optim_method = '$method'" \
    --gin_bindings "Config.pose_alpha0 = 0.6" \
    --gin_bindings "Config.pose_alpha_linear = False" \
    --gin_bindings "Config.pose_render_train = False" \
    --gin_bindings "Config.pose_exam_id = $experiment_count" \
    --logtostderr \
  # 更新实验计数器
  experiment_count=$((experiment_count + 1))
}

# 初始化实验计数器
experiment_count=13

lr_init_values=(0.01)
lr_final_values=(0.006)
optim_values=("manifold")

len=${#lr_init_values[@]}

for (( i=0; i<$len; i++ )); do
  for method in "${optim_values[@]}"; do
    run_experiment ${lr_init_values[$i]} ${lr_final_values[$i]} $method
  done
done