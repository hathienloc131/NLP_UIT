


class Argument():
    def __init__(self):
        self.cuda = False
        self.model_dir = "models"
        self.embedding_size = 768
        self.num_head = 3
        self.cross_attention = True
        self.lr_rate = 1e-4
        self.clip_grad_norm = 1
        self.epochs = 10
        self.batch_size = 10
        self.dropout_rate = 0.3
        self.verdict_size = 3
        self.train_data_path = "data/train.json"
        self.dev_data_path = "data/dev.json"
        self.test_data_path = "data/ise-dsc01-public-test-offcial.json"

