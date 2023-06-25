import torch

VIDEO_EXT = ('.avi', '.gif', '.mp4')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
VIDEO_CACHE_SIZE = 30000
SANITY_CHECK_SIZE = 500

IGNORE_INDEX = -100
