export CUDA_VISIBLE_DEVICES=1
for pred_len in 1 12 24 72 120; do
  python run.py \
    --is_training 0 \
    --seq_len 720 \
    --label_len 48 \
    --pred_len $pred_len \
    --model_id test1 \
    --model CycleNet \
    --data solar_data \
    --root_path ./dataset/Solar_Power/ \
    --data_path solar_data.xlsx \
    --features MS \
    --target "data" \
    --enc_in 12 \
    --batch_size 16 \
    --period_len 12 \
    --twice_epoch 1 \
    --j 1 \
    --pd_ff 256 \
    --pe_layers 1 \
    --adaptive_norm 1 \
    --itr 1
  done