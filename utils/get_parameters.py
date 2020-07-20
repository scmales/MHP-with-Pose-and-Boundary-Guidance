from torch import nn
import collections

def get_parameters(model, predicate):
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)


def load_state(net, checkpoint):   # only weight
    source_state = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    net.load_state_dict(new_target_state)
