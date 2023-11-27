```
# Training commands
python scripts/train.py --terrain single_path --measure_front_half --camera_zero --headless --old_ppo --name vel_empty --empty --penalty_scaler 1.0 --timestep_in_obs --strategy vel --device 2 --terminal_body_height 0.0 --learning_rate 0.0005 --r_base_height 20.0 --r_collision 0.2

python scripts/train.py --terrain single_path --measure_front_half --camera_zero --headless --old_ppo --name vel_single_path --penalty_scaler 1.0 --timestep_in_obs --strategy vel --device 3 --terminal_body_height 0.0 --learning_rate 0.0005 --r_base_height 20.0 --r_collision 0.2

python scripts/train.py --terrain plane --measure_front_half --camera_zero --headless --old_ppo --name vel_empty_terminate_body_height_random_target --penalty_scaler 1.0 --r_task 1.0 --timestep_in_obs --strategy vel --device 2 --terminal_body_height 0.1 --learning_rate 0.0005 --r_base_height 30.0 --r_collision 0.2 --random_target
```

```
# Delete all .mp4 files
find wandb/* -type f -name '*.mp4' -print -delete
```