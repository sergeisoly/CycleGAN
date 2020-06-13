import matplotlib.pyplot as plt
from Ipython.display import clear_output
import numpy as np
import torch

def weights_init_normal(layer): 
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.normal(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.normal(layer.weight.data, 1.0, 0.02)
        torch.nn.init.constant(layer.bias.data, 0.0)

def generate_images(model, test_input, device):
    test_input = test_input.to(device)
    prediction = model(test_input)
    
    plt.figure(figsize=(12, 12))
    test_input = test_input.detach().cpu().squeeze(0).permute(1, 2, 0)
    prediction = prediction[0].detach().cpu().permute(1, 2, 0)
    display_list = [test_input, prediction]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def lambda_lr(epoch):
    # Linear decay
    return 1.0 - max(0, epoch - DECAY_START_EPOCH)/(N_EPOCHS - DECAY_START_EPOCH)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_history(history):
    clear_output(wait=True)
    gen_loss1, disc_A_loss, gen_loss2, disc_B_loss = zip(*history)
    plt.subplot(121)
    plt.title("forward loss")
    plt.plot(moving_average(gen_loss1, n=5), label='gen loss forward')
    plt.plot(moving_average(disc_A_loss, n=5), label='disc loss A')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.title("backward loss")
    plt.plot(moving_average(gen_loss2, n=5), label='gen loss backward')
    plt.plot(moving_average(disc_B_loss, n=5), label='disc loss B')
    plt.legend(loc='best')


def save_models(path, **nets):
    for net in nets.keys():
        torch.save(nets[net].state_dict(), path + f"/{net}.pth")

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad