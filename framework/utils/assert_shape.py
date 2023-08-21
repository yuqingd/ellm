import torch

def assert_shape(tensor, gt_shape):
    """
    credits: https://github.com/ischlag/fast-weight-transformers/blob/main/synthetic/lib.py#L8
    Checks the shape of the tensor for better code redability and bug prevention.
    """
    assert isinstance(tensor, torch.Tensor), "ASSERT SHAPE: tensor is not torch.Tensor!"
    tensor_shape = list(tensor.shape)
    assert len(gt_shape) == len(tensor_shape), f"ASSERT SHAPE: tensor shape {tensor_shape} not the same as {gt_shape}"

    for i, (a,b) in enumerate(zip(tensor_shape, gt_shape)):
        if b <= 0:
            continue # ignore -1 sizes
        else:
            assert a == b, f"ASSERT SHAPE: at idx {str(i)}, tensor shape {tensor_shape} does not match {gt_shape}"