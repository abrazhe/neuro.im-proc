import numpy as np

def read_pic(name):
    fid = open(name, 'rb')
    nx,ny,nz = map(int, np.fromfile(fid, np.uint16, 3))
    start_frames = 0x4c
    fid.seek(start_frames,0)
    frames = np.fromfile(fid, np.uint8, nx*ny*nz).reshape(nz,nx,ny)
    
    meta_start = start_frames + nx*ny*nz + 0x10
    meta = load_meta(fid, meta_start)
    return frames, meta

def load_meta(fid, meta_start, nread=38):
    acc = []
    step = 0x60
    fid.seek(meta_start,0)
    for k in range(0,nread):
        entry = str(fid.read(0x30).strip(b'\x00'))
        acc.append(entry)
        fid.seek(0x30,1)
    return acc

def get_axes(meta):
    ax_info = [e for e in meta if 'axis' in e.lower() and 'microns' in e.lower()]
    acc = []
    for ax in ax_info:
        x = ax.split()[-2:]
        acc.append((float(x[0]), x[1]))
    return acc
