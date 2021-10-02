for NOISE_TYPE in MetroSubway_7; do
    python main.py --task train --method DAT \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 2 --use_source_noise
done