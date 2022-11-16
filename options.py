from custom_types import *                 #改路径就是看见【f' '】就去替换它？！？！？！？！？！
import os                            #在哪里定义obj的获取路径呢？？？？？？？？？？？？？？？？
import pickle
import json
import constants as const
import argparse
import sys
from functools import reduce


class Options:

    @property                        #添加@property后，这个方法（函数）就变成了一个属性 https://zhuanlan.zhihu.com/p/64487092
    def name(self) -> str:
        return f'{self.mesh_name}_{self.tag}' #表明函数name的返回值（return）是字符串类型str 
                                  #f''时，大括号内的所有变量都会被读取并替换为那里的值
                                  #{self.mesh_name}_{self.tag}=mesh_name_tag=cloud_demo(源代码设置的)       所以name=basket_try1 
    
#把class类比作造房子的图纸，那么class实例化后的对象是真正可以住的房子。
#根据一张图纸（类），我们可以设计出成千上万的房子（类对象），每个房子长相都是类似的（都有相同的类变量和类方法），但它们都有各自的主人，那么如何对它们进行区分呢？
#就是各自的self变量，它就相当于每个房子的门钥匙，每个类对象只能调用自己的类变量和类方法。


    @property
    def cp_folder(self) -> str:                             
        return f'{const.PROJECT_ROOT}/dgts_checkpoints/{self.name}'       #在try里面试了一下{self.name}=name自身表示的字符串=basket_try1                                                                               

    @property
    def save_path(self) -> str:                             #这是干嘛的,pkl文件？？？？？？？？？？？
        return f'{self.cp_folder}/options.pkl'

    @property
    def already_saved(self) -> bool:                               #它是干嘛的？布尔值？  and条件需左右同时满足才为true
        return os.path.isfile(self.save_path) and 'debug' not in self.name    #os.path.isfile()检查指定的路径是否已经存在常规文件
                                                        #self.name是？怎么会有debug字符串？？？？？？？？？？？

    @property
    def in_nf(self):                                         
        return 4 * (self.nb_features * 2 + 1)                       #nb_features是啥？？？？？？？

    @property
    def debug(self):
        return 'debug' in self.tag.lower()                         #self.tag.lower？？？？？？

    @property
    def out_nf(self):
        return 3

    def items(self) -> Iterator[str]:                             #迭代器(Iterator)是访问集合内元素的一种方式，提供了遍历序列对象的方法。
        return filter(lambda a: not a.startswith('__') and not callable(getattr(self, a)), dir(self))

    def as_dict(self) -> dict:
        return {item: getattr(self, item) for item in self.items()}

    def save(self):
        if os.path.isdir(self.cp_folder) and not self.already_saved:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            with open(self.save_path[:-3] + 'json', 'w') as f:
                f.write(json.dumps(self.as_dict(), sort_keys=True, indent=4))

    def load(self):
        if self.already_saved:
            with open(self.save_path, 'rb') as f:
                loaded = pickle.load(f)
                print(f'loading options from {self.save_path}')
            return backward_compatibility(loaded)
        return backward_compatibility(self)

    def fill_args(self, **kwargs):
        for arg in kwargs:
            if hasattr(self, arg):
                setattr(self, arg, kwargs[arg])

    @staticmethod
    def in_notebook() -> bool:
        if len(sys.argv) < 2:
            return False
        return reduce(lambda x, y: x or y, map(lambda x: 'jupyter' in x, sys.argv[1:]))

    def parse_cmdline(self):
        if self.in_notebook():
            return
        parser = argparse.ArgumentParser(description='DGTS options')
        parser.add_argument('--tag', type=str, help='')
        parser.add_argument('--mesh-name', type=str, help='')
        parser.add_argument('--template-name', type=str, default='sphere', help='')
        parser.add_argument('--num-levels', type=int, help='')
        parser.add_argument('--start-level', type=int, default=0, help='')
        # inference options
        parser.add_argument('--gen-mode', type=str, choices=['generate', 'animate'])
        parser.add_argument('--num-gen-samples', type=int, default=8)
        parser.add_argument('--target', type=str, default='fertility_al', help='')
        parser.add_argument('--gen-levels', nargs='+', type=int, default=[1, 4], help='')
        # gt optimization options
        parser.add_argument('--template-start', type=int, default=0, help='')

        parser = parser.parse_args().__dict__
        args = {key: item for key, item in parser.items() if item is not None}
        self.fill_args(**args)

    def __init__(self, **kwargs):
        self.tag = 'demo'                                      #主cell里面重新定义了，这里要改嘛？？？？？？？？？？??
        self.mesh_name = 'sphere_rail'
        self.template_name = 'sphere'
        self.start_level = 0
        self.num_levels = 5
        self.start_nf, self.min_nf, self.max_nf, = 32, 32, 128
        self.num_layers = 7
        self.scale_vs_factor = 2
        self.noise_amplitude = 0.1
        self.noise_before = True
        self.update_axes = False
        self.nb_features = False
        self.fix_vs_noise = True
        self.gen_mode, self.num_gen_samples, self.target, self.gen_levels = None, None, None, None
        self.fill_args(**kwargs)


class TrainOption(Options):

    def fill_args(self, **kwargs):
        super(TrainOption, self).fill_args(**kwargs)
        self.level_iters = [2000] * (self.num_levels + self.start_level)

    def __init__(self, **kwargs):
        super(TrainOption, self).__init__()
        self.lr = 5e-4
        self.betas = (.5, .99)
        self.lr_decay = 0.5
        self.lr_decay_every = 500
        self.export_meshes_every = 400
        self.discriminator_iters = 2
        self.generator_iters = 3
        self.reconstruction_weight = 5
        self.penalty_weight = 0.1
        self.inside_out = False
        self.level_iters = [2000] * (self.num_levels + self.start_level)
        self.reconstruction_start = 1
        self.fill_args(**kwargs)


class GtOptions(TrainOption):

    def fill_args(self, **kwargs):
        super(TrainOption, self).fill_args(**kwargs)
        self.level_iters = [3000] * (max(self.num_levels, 1) + self.start_level)
        self.num_samples = [3000 * (i + 1) for i in range(max(self.num_levels, 1) + self.start_level)]

    def __init__(self, **kwargs):
        super(GtOptions, self).__init__()
        self.template_start = 0
        self.lr = 1e-4
        self.gamma_edge_global = 5e-3
        self.gamma_edge_local = 5e-4
        self.gamma_gravity = 0.01
        self.gamma_noraml_nb = 0
        self.gamma_distance_s2t = 0.4
        self.gamma_distance_t2s = 0.6
        self.gamma_noraml_s2t = 0.1
        self.gamma_noraml_t2s = 0.1
        self.ch_iters = 2
        self.triangulation_iters = 3
        self.parse_cmdline()
        self.pre_template = True
        self.switches = ()
        self.fill_args(**kwargs)
        self.level_iters = [3000] * (max(self.num_levels, 1) + self.start_level)
        self.num_samples = [3000 * (i + 1) for i in range(max(self.num_levels, 1) + self.start_level)]

    @property
    def cp_folder(self) -> str:
        return f'{const.DATA_ROOT}/{self.mesh_name}/'   #所以他自己新建了一个covid文件夹在 /dgts_dataset

    @property
    def triangulation_weights(self):
        return self.gamma_edge_global, self.gamma_edge_local, self.gamma_gravity

    @property
    def chamfer_weights(self):
        return self.gamma_distance_s2t, self.gamma_distance_t2s, self.gamma_noraml_s2t, self.gamma_noraml_t2s


def backward_compatibility(opt: Options) -> Options:
    defaults = {}
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    return opt


def copy(opt: Options) -> Options:
    opt_copy = opt.__class__()
    for item in opt_copy.items():
        if hasattr(opt, item):
            try:
                setattr(opt_copy, item, getattr(opt, item))
            except:
                continue
    return opt_copy