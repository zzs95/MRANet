import os
import pickle
import json
import shutil
from copy import deepcopy
def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def findallfiles(base, sort=True):
    res = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            res.append(fullname)

    if sort:
        res.sort()
    return res


subfolders = subdirs  # I am tired of confusing those


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def copyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath, fname = os.path.split(srcfile)
        f_target_path, f_target_name = os.path.split(dstpath)
        if '.' in f_target_name:
            dstpath = f_target_path
            fname = f_target_name
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, join(dstpath, fname))
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))

def filterfile(files_list, pass_keys=[]):
    wanted_files = deepcopy(files_list)
    for f in files_list:
        for not_include in pass_keys:
            if not_include in f:
                try:
                    wanted_files.remove(f)
                    continue
                except:
                    pass
    return wanted_files

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)


save_pickle = write_pickle


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


write_json = save_json


def pardir(path):
    return os.path.join(path, os.pardir)


# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir

'''
Convert a pkl file into json file
'''
import sys
import os
import pickle
import json
import numpy



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl,encoding='latin1')
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4,cls=NumpyEncoder)

