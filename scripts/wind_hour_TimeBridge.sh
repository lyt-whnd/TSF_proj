
model_name=TimeBridge
seq_len=96
GPU=1


alpha=0.35
for pred_len in 1 12 24 72 120
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv\
    --model_id wind_1hour_multi_domain_no_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 17 \
    --ca_layers 1 \
    --pd_layers 2 \
    --ia_layers 4 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 256 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 3 \
    --adaptive_norm 0 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done


model_name=TimeBridge
seq_len=192


alpha=0.35
for pred_len in 1 12 24 72 120
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv\
    --model_id wind_1hour_multi_domain_no_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 17 \
    --ca_layers 1 \
    --pd_layers 2 \
    --ia_layers 4 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 256 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 3 \
    --adaptive_norm 0 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

model_name=TimeBridge
seq_len=336


alpha=0.35
for pred_len in 1 12 24 72 120
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv\
    --model_id wind_1hour_multi_domain_no_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 17 \
    --ca_layers 1 \
    --pd_layers 2 \
    --ia_layers 4 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 256 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 3 \
    --adaptive_norm 0 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done


model_name=TimeBridge
seq_len=512


alpha=0.35
for pred_len in 1 12 24 72 120
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv\
    --model_id wind_1hour_multi_domain_no_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 17 \
    --ca_layers 1 \
    --pd_layers 2 \
    --ia_layers 4 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 256 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 3 \
    --adaptive_norm 0 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

model_name=TimeBridge
seq_len=720


alpha=0.35
for pred_len in 1 12 24 72 120
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv\
    --model_id wind_1hour_multi_domain_no_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 17 \
    --ca_layers 1 \
    --pd_layers 2 \
    --ia_layers 4 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 256 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 3 \
    --adaptive_norm 0 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done