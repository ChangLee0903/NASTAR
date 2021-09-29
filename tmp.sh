for NOISE_TYPE in MetroSubway_7; do
    python main.py --task train --method NASTAR_A09_K250 \
    --target_noise target_data/${NOISE_TYPE}/pseudo.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --cohort_list target_data/${NOISE_TYPE}/cohort.txt \
    --device 2 --use_source_noise --alpha 0.9
done