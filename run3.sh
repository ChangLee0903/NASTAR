for NOISE_TYPE in Car_7 MetroSubway_7; do
    python main.py --task train --method ALL_A09 \
    --target_noise target_data/${NOISE_TYPE}/pseudo.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 2 --use_source_noise --alpha 0.9
done