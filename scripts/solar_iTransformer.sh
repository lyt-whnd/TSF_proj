export CUDA_VISIBLE_DEVICES=1

features=MS
model_name=iTransformer

for pred_len in 1 12 24 72 120; do
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data \
    --model $model_name \
    --data solar_data \
    --features $features \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_ff 256 \
    --d_model 640 \
    --e_layers 4 \
    --des 'Exp' \
    --learning_rate 0.0013163955309262476 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --itr 1 \
    --adaptive_norm 1
  done


for pred_len in 1 12 24 72 120; do
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data \
    --model $model_name \
    --data solar_data \
    --features $features \
    --seq_len 192 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_ff 256 \
    --d_model 640 \
    --e_layers 4 \
    --des 'Exp' \
    --learning_rate 0.0013163955309262476 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --itr 1 \
    --adaptive_norm 1
  done


for pred_len in 1 12 24 72 120; do
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data \
    --model $model_name \
    --data solar_data \
    --features $features \
    --seq_len 336 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_ff 256 \
    --d_model 640 \
    --e_layers 4 \
    --des 'Exp' \
    --learning_rate 0.0013163955309262476 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --itr 1 \
    --adaptive_norm 1
  done


for pred_len in 1 12 24 72 120; do
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data \
    --model $model_name \
    --data solar_data \
    --features $features \
    --seq_len 512 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_ff 256 \
    --d_model 640 \
    --e_layers 4 \
    --des 'Exp' \
    --learning_rate 0.0013163955309262476 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --itr 1 \
    --adaptive_norm 1
  done

for pred_len in 1 12 24 72 120; do
  python -u run.py \
    --is_training 0 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data \
    --model $model_name \
    --data solar_data \
    --features $features \
    --seq_len 720 \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_ff 256 \
    --d_model 640 \
    --e_layers 4 \
    --des 'Exp' \
    --learning_rate 0.0013163955309262476 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --itr 1 \
    --adaptive_norm 1
  done