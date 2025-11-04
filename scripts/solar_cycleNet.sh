export CUDA_VISIBLE_DEVICES=1

python run.py \
--is_training 1 \
--model_id test1 \
--model CycleNet \
--data solar_data \
--root_path ./dataset/electricity/ \
--data_path solar_data.xlsx \
--features MS \
--target "data" \
--enc_in 12 \
--period_len 6 \
--station_lr 0.001 \
--adaptive_norm 1