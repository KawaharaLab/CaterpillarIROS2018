from caterpillar_trainer import train_caterpillar
import multiprocessing
from optparse import OptionParser
from importlib import import_module
import shutil
import os


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--module", dest="module_name")
    parser.add_option("--actor", dest="actor_name")
    parser.add_option("--save", dest="save_dir")
    parser.add_option("--times", default=10, dest="train_times", type='int')
    parser.add_option("-d", action="append", dest="disable_list", default=[], type=int)
    opts, args = parser.parse_args()

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    multiprocessing.set_start_method('spawn')   # Tensorflow is not fork safe.

    if opts.disable_list != []:
        print("force sensor disable list", opts.disable_list)

    for trial in range(opts.train_times):
        print("Init position training... (trial {})".format(trial))
        trial_save_dir = os.path.join(opts.save_dir, "trial{}".format(trial))
        if os.path.exists(trial_save_dir):
            shutil.rmtree(trial_save_dir)

        train_caterpillar(trial_save_dir, actor_module_name=opts.module_name, actor_name=opts.actor_name, disable_list=opts.disable_list)
        print("Result saved in {}".format(trial_save_dir))
