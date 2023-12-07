import torch
from netCDF4 import Dataset
import numpy as np


def make_coord_grid(shape, device, flatten=True, align_corners=False, use_half=False):
    """ 
    Make coordinates at grid centers.
    return (shape.prod, 3) matrix with (z,y,x) coordinate
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        left = -1.0
        right = 1.0
        if(align_corners):
            r = (right - left) / (n-1)
            seq = left + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()

        else:
            r = (right - left) / (n+1)
            seq :torch.Tensor = left + r + r * \
            torch.arange(0, n, 
            device=device, 
            dtype=torch.float32).float()
            
        if(use_half):
                seq = seq.half()
        coord_seqs.append(seq)

    ret = torch.meshgrid(*coord_seqs, indexing="ij")
    ret = torch.stack(ret, dim=-1)
    if(flatten):
        ret = ret.view(-1, ret.shape[-1])
    return ret.flip(-1)

def tensor_to_cdf(t, location, channel_names=None):
    # Assumes t is a tensor with shape (1, c, d, h[, w])

    d = Dataset(location, 'w')

    # Setup dimensions
    d.createDimension('x')
    d.createDimension('y')
    dims = ['x', 'y']

    if(len(t.shape) == 5):
        d.createDimension('z')
        dims.append('z')

    # ['u', 'v', 'w']
    if(channel_names is None):
        ch_default = 'a'

    for i in range(t.shape[1]):
        if(channel_names is None):
            ch = ch_default
            ch_default = chr(ord(ch)+1)
        else:
            ch = channel_names[i]
        d.createVariable(ch, np.float32, dims)
        d[ch][:] = t[0,i].detach().cpu().numpy()
    d.close()

f = 8
g = make_coord_grid([128, 128, 128], "cuda", True)*2

real = torch.exp(-(torch.norm(g, dim=1)**2))*(0.5+0.5*torch.cos(g[:,0]*f))
real = real.reshape([128, 128, 128])[None,None,...]
tensor_to_cdf(real, f"gabor{f}_real.nc")

result = torch.zeros([g.shape[0]], device="cuda", dtype=torch.float32)
gaussians_on_each_size = int(np.round((2*f/np.pi - 1) / 2))
wavelength = (2*np.pi)/f
print(gaussians_on_each_size)
for k in range(-gaussians_on_each_size,gaussians_on_each_size+1):    
    center = torch.tensor([k*wavelength, 0., 0.], device="cuda")
    diff = g-center[None,...]
    diff[:,0] *= f/1.618
    gauss = np.exp(-((k*2*torch.pi/f)**2))*torch.exp(-(torch.norm(diff, dim=1)**2))
    result += gauss

result = result.reshape([128, 128, 128])[None,None,...]
tensor_to_cdf(result, f"gabor{f}_approx.nc")