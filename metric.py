import torch

class TrainDiscriminatorMetric():
    def __init__(self, fake_idx):
        self.acc_all = []
        self.acc_fake = []
        self.acc_real = []
        self.loss = []
        self.fake_idx = fake_idx

    def __call__(self, logits, label, loss):
        self.loss.append(loss)
        logits = (torch.sigmoid(logits) > 0.5)
        label = (label > 0.5)
        res = (logits == label).float()
        acc = torch.mean(res).item()
        self.acc_all.append(acc)
        real_idx = label[:, self.fake_idx] == False
        fake_idx = label[:, self.fake_idx] == True
        real_label = label[real_idx, :]
        fake_label = label[fake_idx, :]
        real_logits = logits[real_idx, :]
        fake_logits = logits[fake_idx, :]
        self.acc_fake.append(torch.mean((fake_logits == fake_label).float()).item())
        self.acc_real.append(torch.mean((real_logits == real_label).float()).item())

    def value(self):
        num = len(self.acc_all)
        return sum(self.acc_all) / num, sum(self.acc_fake) / num, sum(self.acc_real) / num, sum(self.loss) / num

    def reset(self):
        self.acc_all.clear()
        self.acc_fake.clear()
        self.acc_real.clear()
        self.loss.clear()
