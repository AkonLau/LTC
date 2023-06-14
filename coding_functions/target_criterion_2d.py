import torch
import torch.nn as nn

import random
from scipy.linalg import hadamard

class Sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_o):
        return grad_o.clamp_(-1, 1)

class LearnableTargetCoding(nn.Module):
    def __init__(self, gamma_=1,
                 lambda_=0.01,
                 beta_=0.1,
                 code_length=512,
                 classes_num=200,
                 active_type='sgn',
                 margin_ratio=1,
                 attr_type='mse',
                 ):
        super(LearnableTargetCoding, self).__init__()
        self.target_labels = nn.Parameter(torch.randn(classes_num, code_length))

        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.beta_ = beta_
        self.classes_num = classes_num
        self.margin = code_length * margin_ratio

        self.active_type = active_type
        if self.active_type == 'sgn':
            self.target_activation = Sign.apply
        elif self.active_type == 'tanh':
            self.target_activation = nn.Tanh()
        else:
            raise NameError("WARN: The target_activation '{}' is not supported here.".format(self.active_type))

        self.attr_type = attr_type
        if self.attr_type == 'mse':
            self.Loss = nn.MSELoss()
            self.activation = nn.Tanh()
        else:
            raise NameError("WARN: The loss type '{}' is not supported here.".format(self.attr_type))

    def print_target_codes(self):
        return self.target_activation(self.target_labels)

    def forward(self, xlocal, targets):
        xlocal = self.activation(xlocal)
        batch_size = xlocal.size(0)

        attr_values = self.target_activation(self.target_labels)
        target_label = attr_values[targets]
        
        # mse_loss
        reg_loss = self.Loss(xlocal, target_label) * self.gamma_
        if self.lambda_ > 0 :
            # triplet_loss
            dist_matrix = xlocal.mm(attr_values.T) # B x C

            index = torch.arange(batch_size)
            positive_dist = dist_matrix[index, targets].unsqueeze(1).repeat(1, self.classes_num)

            mask = torch.ones_like(dist_matrix)
            mask = torch.scatter(mask, 1, targets.unsqueeze(1), torch.zeros_like(targets.unsqueeze(1), dtype=torch.float32))

            triplet_loss_arr = (dist_matrix - positive_dist + self.margin) * mask
            index4mean = (triplet_loss_arr > 0)
            num_index  = (triplet_loss_arr > 0).sum()
            if num_index > 0 :
                triplet_loss = triplet_loss_arr[index4mean].mean()
            else:
                triplet_loss = torch.tensor(0).cuda()

            reg_loss += (triplet_loss * self.lambda_)
        if self.beta_ > 0:
            #  corr_loss
            attr_dist_matrix = attr_values.mm(attr_values.T) # C x C
            attr_mask = 1 - torch.eye(self.classes_num).cuda()
            corr_loss = (attr_dist_matrix * attr_mask).abs().mean()

            reg_loss += (corr_loss * self.beta_)
            
        return reg_loss

class HadamardTargetCoding(nn.Module):
    def __init__(self, 
                 gamma_=1, 
                 code_length=512, 
                 classes_num=200,
                 attr_type='mse',
                 ):
        super(HadamardTargetCoding, self).__init__()
        self.target_labels = self._get_attr_values(classes_num=classes_num,
                                           target_code_length=code_length)
        self.gamma_ = gamma_
        self.attr_type = attr_type

        if self.attr_type == 'mse':
            self.Loss = nn.MSELoss()
            self.activation = nn.Tanh()
            self.target_labels = self.target_labels # [-1, 1]
        else:
            raise NameError("WARN: The loss type '{}' is not supported here.".format(self.attr_type))

    def _get_attr_values(self, classes_num, target_code_length):
        classes_ha_d = hadamard(target_code_length)[1:]
        class_index = random.sample(list(range(target_code_length - 1)), classes_num)
        class_code = torch.from_numpy(classes_ha_d[class_index]).float().cuda()
        return class_code

    def print_target_codes(self):
        return self.target_labels

    def forward(self, xlocal, targets):
        xlocal = self.activation(xlocal)
        target_label = self.target_labels[targets]
        reg_loss = self.Loss(xlocal, target_label) * self.gamma_

        return reg_loss


if __name__ == '__main__':
    code_length = 512
    classes_num = 200

    xlocal = torch.randn(128, code_length).cuda()
    xlocal = nn.Tanh()(xlocal)
    targets = torch.LongTensor(128).random_(0, classes_num).cuda()

    ltc_criterion = LearnableTargetCoding().cuda()
    htc_criterion = HadamardTargetCoding().cuda()
    for i in range(0, 5):
        ltc_loss = ltc_criterion(xlocal, targets)
        htc_loss = htc_criterion(xlocal, targets)
        print('ltc_loss:', ltc_loss, 'htc_loss:', htc_loss)
