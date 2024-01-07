import torch

# loss_type: ['reg_loss', 'class_loss', 'regular_loss']
def get_loss(loss_type, loss_name):
    if loss_type == 'reg_loss':
        loss_func = getattr(RegLoss, loss_name)
    return loss_func


class RegLoss:
    @staticmethod
    def default(output, label):
        mean_loss = torch.mean(torch.pow(torch.abs(output-label), 1))
        max_loss = torch.max(torch.pow(torch.abs(output-label), 1))
        loss = mean_loss + max_loss
        return loss