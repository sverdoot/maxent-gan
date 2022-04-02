import torch


def holder_hook(holder):
    """
    TODO
    """
    def hook(model, input, output):
        holder.append(output)

    return hook


def penult_layer_activation(model, input: torch.Tensor):
    """
    TODO
    """
    activation = []
    hook = model.penult_layer.register_forward_hook(holder_hook(activation))
    model(input)
    out = torch.cat([_.to(input.device) for _ in activation], 0).view(len(input), -1)
    activation = []
    hook.remove()
    return out
