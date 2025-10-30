if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TimeBridge" ]; then
    mkdir ./logs/LongForecasting/TimeBridge
fi

model_name=TimeBridge
seq_len=720
GPU=1


alpha=0.35
data_name=solar_data
for pred_len in 96
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --model_id solar_data_96_96 \
    --model $model_name \
    --data solar_data \
    --features MS \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 12 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --target 'data' \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 | tee logs/LongForecasting/TimeBridge/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done