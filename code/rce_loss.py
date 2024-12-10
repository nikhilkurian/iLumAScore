'''Based on
Kurian, Nikhil Cherian, et al. "Sample specific generalized cross entropy for robust histology image classification." 
2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI). IEEE, 2021.
'''

import torch
import torch.nn.functional as F
import pandas as pd
from scipy.special import beta as beta_func
from scipy.optimize import fsolve


def SSGCEScore(df):
    min_loss = df['Loss'].min()
    max_loss = df['Loss'].max()
    df['Normalized_Loss'] = (df['Loss'] - min_loss) / (max_loss - min_loss)
    mean_loss = df['Normalized_Loss'].mean()
    var_loss = df['Normalized_Loss'].var()

    a = mean_loss * ((mean_loss * (1 - mean_loss) / var_loss) - 1)
    b = (1 - mean_loss) * ((mean_loss * (1 - mean_loss) / var_loss) - 1)

    def find_gamma(a, b, target_expectation=0.7):
        def equation(gamma):
            beta_dist = beta(a, b)
            expected_value = beta_dist.expect(lambda x: x**gamma)
            return expected_value - target_expectation

        gamma_initial_guess = 1.0 
        gamma_solution = fsolve(equation, gamma_initial_guess)[0]
        return gamma_solution

    gamma = find_gamma(a, b)
    df['q'] = df['Normalized_Loss'] ** gamma

    return df


class SSGCELoss(torch.nn.Module):
    def __init__(self):
        super(SSGCELoss,self).__init__()

    def forward(self, y_pred, y_true, _q=0.7):
        y_pred=F.softmax(y_pred,dim=1)
        _tmp = y_pred * y_true
        _loss,_ = torch.max(_tmp, -1)
        _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q
        _loss = torch.mean(_loss)
        return _loss



