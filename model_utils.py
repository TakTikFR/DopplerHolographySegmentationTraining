import torch
from huggingface_hub import upload_file
import onnx
import onnxruntime as ort
import torch.nn as nn
import time

def export_jit(model, path, input_shape=(1, 1, 512, 512)):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(path)

def export_onnx(model, path, input_shape=(1, 1, 512, 512)):
    model.eval()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model.cpu(), dummy_input, path, 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

def load_onnx_model(path, device='cuda'):
    # Load the ONNX model
    model = onnx.load(path)
    
    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Create an ONNX Runtime session
    session = ort.InferenceSession(path, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])

    return session

def _get_first_layer(m):
    "Access first layer of a model"
    c,p,n = m,None,None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p,c=c,getattr(c,n)
    return c,p,n

def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    print(f"Previous layer weights shape: {previous_layer.weight.data.shape}, new layer weights shape: {new_layer.weight.data.shape}")
    if n_in==1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
        print(f"Previous layer weights shape: {previous_layer.weight.data.shape}, new layer weights shape: {new_layer.weight.data.shape}")
    elif n_in==2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:2,:] * 1.5
    else:
        # keep 3 channels weights and set others to null
        print(f"Warning: More than 3 input channels, only first 3 channels will be initialized with pretrained weights, others will be set to zero.")
        new_layer.weight.data[:2,:] = previous_layer.weight[:2,:].data
        new_layer.weight.data[2:,:].zero_()

def _update_conv_layer(layer, param_str, param, pretrained):
    "Change layer based on parameter"
    assert isinstance(layer, nn.Conv2d), f'Change only supported with Conv2d, found {layer.__class__.__name__}'
    params = {attr:getattr(layer, attr) for attr in 'in_channels out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(layer, 'bias') is not None
    params[param_str] = param
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, layer)
    return new_layer

def _update_first_layer_input(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    new_layer = _update_conv_layer(first_layer, 'in_channels', n_in, pretrained)
    setattr(parent, name, new_layer)

def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)

def upload_file_to_hf(repo_id, hf_model_name, file_path):
    repo_id = "DigitalHolography"
    huggingface_model_name = "nnwnet_av_corr_diasys"
    repo_id = f"{repo_id}/{huggingface_model_name}"
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=huggingface_model_name,
        repo_id=repo_id,
        repo_type="model"
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_tensor, device='cuda', iterations=100):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    # Warm-up (important for GPU)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()  # ensure all ops are finished
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations
    print(f"Average inference time per run: {avg_time * 1000:.3f} ms")
    return avg_time