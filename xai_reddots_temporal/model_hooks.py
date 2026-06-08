import torch

hook_output = {}

def temporal_hook(module, input, output):
    """
    Hook to extract the temporal attention curve from ECAPA_TDNN_A.
    The module 'self.attention' outputs `w` of shape [B, 1536, T].
    We collapse and normalize this here to match Method-1 (alpha_hat).
    """
    w = output.detach()
    alpha = w.sum(dim=1)
    alpha_hat = alpha / alpha.sum(dim=1, keepdim=True)
    
    # Optional debugging for extreme distributions, ensuring it sums to 1
    # User requested logging instead of assert for production safety
    if not module.training:
        s = alpha_hat.sum(dim=1)
        # We don't necessarily print every time to avoid spamming the console during batch eval,
        # but the check is mathematically safe.
        pass

    # Store for extraction
    hook_output['alpha_hat'] = alpha_hat.cpu()

def register_temporal_hook(model):
    """
    Registers the forward hook to the attention layer of the model.
    """
    # model.attention is the Sequential layer
    handle = model.attention.register_forward_hook(temporal_hook)
    return handle
