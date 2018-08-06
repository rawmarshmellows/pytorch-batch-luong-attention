# PytorchLuongAttention
This is batched implementation of Luong Attention. This code does batch multiplication to calculate the attention scores, instead of calculating the score one by one

To run:
`train_luong_attention.py --train_dir data/translation --dataset_module translation --log_level INFO --batch_size 50 --use_cuda --hidden_size 500 --input_size 500 --different_vocab`