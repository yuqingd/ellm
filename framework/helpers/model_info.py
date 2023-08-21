import numpy as np


def print_model_size(model):
    line_len = 89
    line_len2 = 25
    print('-' * line_len)
    # Native pytorch
    try:
        print(model)
    except:
        print('Warning: could not print the Native PyTorch model info - probably some module is `None`.')

    # One-by-one layer
    print('-' * line_len)
    print("Model params:")
    total_params = 0
    module_name = ""
    module_n_params = 0
    for name, param in model.named_parameters():
        if module_name == "":
            module_name = name[:name.index('.')]
        if module_name != name[:name.index('.')]:
            print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')
            module_name = name[:name.index('.')]
            module_n_params = 0
        n_params = np.prod(param.size())
        module_n_params += n_params
        print(f"\t {name} {n_params:,}")
        total_params += n_params
    print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')

    # Total Number of params
    print('-' * line_len)
    print(f"Total number of params: {total_params:,}")
    print('-' * line_len)
