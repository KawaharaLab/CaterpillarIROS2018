import multiprocessing
from optparse import OptionParser
import shutil
import os

import caterpillar_trainer
from controllers import utils


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", dest="module_name", help="Module name of actor model to use.")
    parser.add_option("-s", dest="save_dir", help="Path of a directory to save. If a directory with the provided path exists, it is overwritten.")
    parser.add_option("-t", default=10, dest="train_times", type='int', help="Iteration number of training.")
    opts, args = parser.parse_args()

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    multiprocessing.set_start_method('spawn')   # Tensorflow is not fork safe.

    for trial in range(opts.train_times):
        print("Init position training... (trial {})".format(trial))

        trial_save_dir = os.path.join(opts.save_dir, "trial{}".format(trial))
        if os.path.exists(trial_save_dir):
            shutil.rmtree(trial_save_dir)
        save_dir = utils.SaveDir(trial_save_dir)

        caterpillar_trainer.train_caterpillar(save_dir, actor_module_name=opts.module_name)
        print("Result saved in {}".format(trial_save_dir))
