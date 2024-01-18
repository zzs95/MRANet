# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import scipy.ndimage as ndimage

def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np
    import torch


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_file_path):
    # datasets_as_dfs = get_datasets_as_dfs(config_file_path)
    with open(config_file_path, "r") as f:
        for ann in f.readlines():
            ann = ann.strip('\n').strip('	')  # remove
            print(ann)
            try:
                param_name, param_value = ann.split(': ')  # split
                exec('global '+ param_name + ' = '+ param_value) # create variable
            except:
                pass

def write_config(config_file_path, config_parameters):
    with open(config_file_path, "w") as f:
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")
