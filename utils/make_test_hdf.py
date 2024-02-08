import pandas as pd
import numpy as np
from tqdm import tqdm
hdf = pd.read_hdf('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf',  'df')
test_hdf = hdf[:50000]
test_hdf.to_hdf('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116_test.hdf',  'df')


