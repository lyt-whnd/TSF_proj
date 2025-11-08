export CUDA_VISIBLE_DEVICES=1
for pred_len in 1; do
  python run.py \
    --is_training 0 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --model_id test1 \
    --model CycleNet \
    --data solar_data \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data_80a7415eb35e48378ab8220c12aa7327_副本.xlsx \
    --features MS \
    --enc_in 12 \
    --batch_size 64 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --adaptive_norm 1 \
    --itr 1
  done