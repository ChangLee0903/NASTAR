python main.py --task train --method PTN --loss mrstft --model DEMUCS --config config/pretrain.yaml
python main.py --task train --method PTN --loss sisdr --model LSTM --config config/pretrain.yaml
python main.py --task train --method PTN --loss sisdr --model GRU --config config/pretrain.yaml