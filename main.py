from bi_encoder_cross_attention_model.train import train as train_bi_encoder_cross_attention
import sys

if __name__ == '__main__':
    type_model = sys.argv[1]
    match type_model:
        case "1":
            train_bi_encoder_cross_attention()
            pass