export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

model_name=iTransformer
for pred_len in 1 4 12
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Wind_data/ \
    --data_path wind_10min_multi_domain.csv \
    --model_id wind_10min_multi_domain_96_96 \
    --model $model_name \
    --data Wind_multi_domain \
    --features MS \
    --seq_len 1024 \
    --pred_len $pred_len \
    --e_layers 8 \
    --enc_in 17 \
    --dec_in 17 \
    --c_out 17 \
    --des 'Exp' \
    --batch_size 64 \
    --d_model 512\
    --d_ff 512\
    --patience 3\
    --itr 1 | tee logs/LongForecasting/$model_name'VREX_''Wind_multi_domain'$pred_len.log
  done