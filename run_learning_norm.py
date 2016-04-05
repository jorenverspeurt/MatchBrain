from __future__ import print_function

import datetime as dt
import json
import logging
import time
from itertools import repeat

import numpy as np
from keras import backend as K
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from ml.learning import AutoTransformer
from ml.logutils import NormSourceMaker

#LOG
logger = logging.getLogger('learning')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def toStr():
    def decorator(f):
        class _temp:
            def __call__(self, *args, **kwargs):
                return f(*args, **kwargs)
            def __str__(self):
                return f.__name__
        return _temp()
    return decorator

class nest(SGD):
    def __init__(self, lr, momentum):
        SGD.__init__(self, lr=lr, decay=1e-6, momentum=momentum, nesterov=True)

    def __str__(self):
        return "nest"+str(self.lr)+"_"+str(self.momentum)

#DEFS
phase_names = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']
label_normalization = [0.82967276, 1.69463687, 1.74141838, 0.3860981,  0.34817388]
@toStr()
def biased_cce(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, label_normalization * y_true), axis=-1)

time_str = lambda: dt.datetime.now().strftime('%y%m%d-%H%M%S')

encdec_opt_strings = ["drop_rate", "enc_use_drop", "enc_use_noize", "encdec_optimizer", "epochs", "gaussian_base_sigma", "gaussian_sigma_factor"]

#PARAMS
optimizers = ['adadelta', 'rmsprop', 'adam', 'adagrad', 'sgd'] + [nest(l,m)
                                                                  for l in [1, 0.1, 0.01]
                                                                  for m in [0.5, 0.9, 0.95, 0.99]]
class_losses = [biased_cce, 'categorical_crossentropy', 'mean_squared_error']
drop_rates = [0, 0.001, 0.01, 0.1]
gauss_base_sigmas = [0.0, 0.001, 0.01, 0.02, 0.05]
gauss_sigma_factors = [1, 1.1, 1.5, 2, 3]
epochs = 600
data_mins = 40

#SPECIFIC
pretrain_encdecs = False
continue_encdecs = False
finetune = False
continue_finetune = False
evaluate_model = False
cross_validate = True

def update_status(ae, key, value):
    with open(ae.model_dir + "status.json", 'r') as f:
        status = json.loads(f.read())
        if key == "history":
            status.setdefault("history", []).insert(0, value)
        else:
            status[key] = value
    with open(ae.model_dir + "status.json", 'w') as f:
        f.write(json.dumps(status, indent=2, sort_keys=True))

if __name__ == '__main__':
    ### SETUP ###
    nsm = NormSourceMaker(datafolder="/home/joren/PycharmProjects/MatchBrain/ml/"
                         ,phases=phase_names
                         ,cross_val=True)
    AutoTransformer.model_dir = "/home/joren/PycharmProjects/MatchBrain/models/"
    for _ in xrange(len(nsm.cross_val_keys)):
        blk = nsm.get_block()
        #TODO get this size from somewhere better. Is it even correct?
        ae = AutoTransformer(100, blk.sinks[0], blk.sinks[1], epochs=epochs, num_sizes=5) #TODO get this 100 from somewhere reliable
        print("have ae")
        blk.start()
        while blk.started and data_mins>0:
            time.sleep(60)
            print(ae.batched)
            data_mins -= 1
        if blk.started:
            blk.stop()
        print("stopped")
        ae.cap_data()
        print("capped")
        ### PRETRAIN ###
        losscomb = lambda zoo, h: ", ".join(map(lambda (i,e): str(i)+":"+('%.4f'%e[-1][zoo]), enumerate(h)))
        n = nest(1,0.9)
        counter = 0
        options = [('adadelta', 0.001, 0.001, 2)]
        ls = [0.001]
        secondary = [(0, l2) for l2 in ls]
        for (opt, dr, gbs, gsf) in options:
            for (l1, l2) in secondary:
                counter += 1
                print(str(counter)+" of "+str(len(options)*len(secondary)))
                name = "o-"+str(opt)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)+"-l1-"+str(l1)+"-l2-"+str(l2)
                ae.enc_opt = opt
                ae.drop_rate = dr
                ae.sigma_base = gbs
                ae.sigma_fact = gsf
                ae.l1 = l1
                ae.l2 = l2
                ae.new_encdecs(use_dropout=(dr != 0), use_noise=(gbs != 0))
                print("training encdec "+name)
                h = ae.pretrain(name=name)
                info = {"acc_pre_"+str(i): float(e[-1][1]) for (i,e) in enumerate(h)}
                info.update({("loss_pre_"+str(i)): float(e[-1][0]) for (i,e) in enumerate(h)})
                ae.update_catalog("ed_"+name,info)
        ### FINETUNE ###
        cur_left_out = nsm.cross_val_keys[nsm.cross_val_index]
        ae.load_encdecs("best")
        ae.finetune(name="x-"+cur_left_out+"-val", train_encdecs=True, test_data=nsm.data[cur_left_out])
        info = {}
        for key in ae.previous_data:
            print(key)
            arr = np.array(ae.previous_data[key])
            phl = len(phase_names)
            eye = np.identity(phl)
            tc = to_categorical
            eva = ae.model.evaluate(arr,
                                    tc(map(lambda n: phase_names.index(n)
                                           ,list(repeat(key, len(arr))))
                                       ,phl)
                                    ,show_accuracy=True)
            counts = np.sum(map(lambda p: eye[np.argmax(p)], ae.model.predict(arr)), axis=0)
            print(eva)
            print(counts)
            info.update({key: [eva,counts]})
        ae.update_catalog(ae.model_name, info)
        nsm.cross_val_next()
