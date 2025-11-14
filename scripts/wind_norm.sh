export CUDA_VISIBLE_DEVICES=1
model_name=iTransformer
#for pred_len in 12 24 72 120
#do
#  python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/Wind_data/ \
#    --data_path output_hourly.csv\
#    --model_id wind_1hour_multi_domain_norm \
#    --model $model_name \
#    --data Wind_multi_domain \
#    --features MS \
#    --seq_len 96 \
#    --pred_len $pred_len \
#    --e_layers 8 \
#    --enc_in 17 \
#    --dec_in 17 \
#    --c_out 17 \
#    --des 'Exp' \
#    --batch_size 64 \
#    --d_model 512 \
#    --d_ff 512 \
#    --patience 3 \
#    --batch_size 64 \
#    --period_len 12 \
#    --kernel_len 16 \
#    --twice_epoch 1 \
#    --j 1 \
#    --pd_ff 256 \
#    --pe_layers 1 \
#    --adaptive_norm 1 \
#    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
#done
#
#
#
#for pred_len in 12 24 72 120
#do
#  python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/Wind_data/ \
#    --data_path output_hourly.csv \
#    --model_id wind_1hour_multi_domain_norm \
#    --model $model_name \
#    --data Wind_multi_domain \
#    --features MS \
#    --seq_len 192 \
#    --pred_len $pred_len \
#    --e_layers 8 \
#    --enc_in 17 \
#    --dec_in 17 \
#    --c_out 17 \
#    --des 'Exp' \
#    --batch_size 64 \
#    --d_model 512 \
#    --d_ff 512 \
#    --patience 3 \
#    --batch_size 64 \
#    --period_len 12 \
#    --kernel_len 16 \
#    --twice_epoch 1 \
#    --j 1 \
#    --pd_ff 256 \
#    --pe_layers 1 \
#    --adaptive_norm 1 \
#    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
#done

for pred_len in 120
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv \
    --model_id wind_1hour_multi_domain_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len 336 \
    --pred_len $pred_len \
    --e_layers 8 \
    --enc_in 17 \
    --dec_in 17 \
    --c_out 17 \
    --des 'Exp' \
    --batch_size 64 \
    --d_model 512 \
    --d_ff 512 \
    --patience 3 \
    --batch_size 64 \
    --period_len 12 \
    --kernel_len 16 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --adaptive_norm 1 \
    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
done

for pred_len in 12 24 72 120
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv \
    --model_id wind_1hour_multi_domain_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len 512 \
    --pred_len $pred_len \
    --e_layers 8 \
    --enc_in 17 \
    --dec_in 17 \
    --c_out 17 \
    --des 'Exp' \
    --batch_size 64 \
    --d_model 512 \
    --d_ff 512 \
    --patience 3 \
    --batch_size 64 \
    --period_len 12 \
    --kernel_len 16 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --adaptive_norm 1 \
    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
done

for pred_len in 12 24 72 120
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path output_hourly.csv \
    --model_id wind_1hour_multi_domain_norm \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len 720 \
    --pred_len $pred_len \
    --e_layers 8 \
    --enc_in 17 \
    --dec_in 17 \
    --c_out 17 \
    --des 'Exp' \
    --batch_size 64 \
    --d_model 512 \
    --d_ff 512 \
    --patience 3 \
    --batch_size 64 \
    --period_len 12 \
    --kernel_len 16 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --adaptive_norm 1 \
    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
done