
model_name=TimeBridge
seq_len=576
GPU=0


alpha=0.35
for pred_len in 6 72 144 432 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain \
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
seq_len=1152


alpha=0.35
for pred_len in 6 72 144 432 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain \
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
seq_len=2016


alpha=0.35
for pred_len in 6 72 144 432 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain \
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
seq_len=3072


alpha=0.35
for pred_len in 6 72 144 432 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain \
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
seq_len=4320


alpha=0.35
for pred_len in 6 72 144 432 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain \
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