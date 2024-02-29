import torch
def cross_entropy(targets_soft, predictions_soft, epsilon = 1e-12):

    #把prediction的值限制在epsilon 和1. -epsilion 之间
    predictions = torch.clamp(predictions_soft, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -torch.sum(targets_soft*torch.log(predictions))/N

    return ce
lables=torch.tensor([[0.1,0.9],[0.5,0.5]])
output=torch.tensor([[0.1,0.9],[0.5,0.5]])
cross_entropy(lables,output)
print(cross_entropy(lables,output))
