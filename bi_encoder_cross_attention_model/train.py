from torch.utils.data import DataLoader
import numpy as np


from utils.analysis import get_analysis
from utils.processing import load_datasets


from .dataset import CustomDataset
from .argument import Argument
from .model import Model


# Dev the model
def dev(model, dev_dataloader):
    evidence_acc_total = []
    verdict_acc_total = []
    loss_total = []
    for i, batch_input in enumerate(dev_dataloader):

        evidence_acc, verdict_acc, loss = model(batch_input)
        print(f'{i}, loss: {loss}')
        evidence_acc_total.extend(evidence_acc.tolist())
        verdict_acc_total.extend(verdict_acc.tolist())
        loss_total.append(loss.item())

    print("evidence_acc_total: ", np.sum(evidence_acc_total)/len(evidence_acc_total))
    print("verdict_acc_total: ", np.sum(verdict_acc_total)/len(verdict_acc_total))
    print("loss_total: ", np.sum(loss_total)/len(loss_total))



def train():
    args = Argument()
    weight_label = get_analysis(args.train_data_path)
    print(weight_label)
    model = Model(args, weight_label)
    if args.cuda:
        model = model.cuda()

    train_data, train_max_sent_len = load_datasets(args.train_data_path)
    # dev_data, dev_max_sent_len = load_datasets(args.dev_data_path)
    #test_data, max_sent_len = load_datasets(args.data_path, is_test=True)
    
    # train_dataset = CustomDataset(train_data, train_max_sent_len)
    # dev_dataset = CustomDataset(dev_data, dev_max_sent_len)

    exit()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    for e in range(args.epochs):
        print("epoch %d" % e)
        for i, batch_input in enumerate(train_dataloader):
            evidence_acc, verdict_acc, loss = model(batch_input)
            model.step(loss)
            print(f'{model.global_step}, loss: {loss}')
            if model.global_step % 1000 == 0:
                model.save_model()
                dev(model, dev_dataloader)

