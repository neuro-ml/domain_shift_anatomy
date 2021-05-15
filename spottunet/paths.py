from .utils import choose_root


DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/cc359',
    '/gpfs/data/gpfs0/b.shirokikh/data/cc359',
)

BASELINE_PATH = choose_root(
    '/gpfs/data/gpfs0/b.shirokikh/experiments/da/miccai2021_spottune/baseline/cc359_unet2d_one2all',
)
