from __future__ import print_function

import json
import logging
import time
from itertools import repeat

import numpy as np
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils

from core.TrainView import phase_names
from ml.learning import AutoTransformer
from ml.logutils import LogSourceMaker
from ml.processing import Preprocessing

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
label_normalization = [0.82967276, 1.69463687, 1.74141838, 0.3860981,  0.34817388]
@toStr()
def biased_cce(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_pred, label_normalization * y_true), axis=-1)

#PARAMS
optimizers = ['adadelta', 'rmsprop', 'adam', 'adagrad', 'sgd'] + [nest(l,m)
                                                                  for l in [1, 0.1, 0.01]
                                                                  for m in [0.5, 0.9, 0.95, 0.99]]
class_losses = [biased_cce, 'categorical_crossentropy', 'mean_squared_error']
drop_rates = [0, 0.001, 0.01, 0.1]
gauss_base_sigmas = [0.0, 0.001, 0.01, 0.02, 0.05]
gauss_sigma_factors = [1, 1.1, 1.5, 2, 3]
epochs = 120
shift = 1

#SPECIFIC
data_mins = 40
generate_data = True
pretrain_encdecs = True
finetune = False

def update_status(ae, key, value):
    f = open(ae.model_dir + "status.json", 'r+')
    status = json.loads(f.read())
    if key == "history":
        status.setdefault("history", []).insert(0, value)
    else:
        status[key] = value
    f.write(json.dumps(status, indent=2, sort_keys=True))
    f.close()

if __name__ == '__main__':
    ### SETUP ###
    l = LogSourceMaker(logfolder="/root/MatchBrain/logs/")
    b = l.get_block(shift = shift)
    bws = b.sources[0]
    phs = b.sources[1]
    prep = Preprocessing(bws)
    print("prep output dim: "+str(prep.output_dim))
    ae = AutoTransformer(prep.output_dim, prep, phs, epochs=120, num_sizes=5)
    ae.model_dir = "/root/MatchBrain/models/"
    print("have ae")
    ### DATA ###
    if generate_data:
        update_status(ae, "current", "preprocessing data")
        b.start()
        while b.started and data_mins>0:
            time.sleep(15)
            print(ae.batched)
            data_mins -= 1
        if b.started:
            b.stop()
        print("stopped")
        update_status(ae, "current", "saving data")
        ae.save_data()
        print("data saved")
    else:
        ae.load_data()
    ### PRETRAIN ###
    if pretrain_encdecs:
        losscomb = lambda zoo, h: ",".join(map(lambda (i,e): str(i)+":"+e[-1][zoo], enumerate(h)))
        n = nest(1,0.9)
        counter = 0
        for opt in ['rmsprop', 'adadelta', nest(1,0.9)]:
            for dr in drop_rates[0:-1]:
                for gbs in gauss_base_sigmas[0:3]:
                    for gsf in [1, 1.1, 2]:
                        counter += 1
                        print(str(counter)+" of "+str(3**4))
                        name = "o-"+str(opt)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)
                        ae.enc_opt = opt
                        ae.drop_rate = dr
                        ae.sigma_base = gbs
                        ae.sigma_fact = gsf
                        ae.new_encdecs(use_dropout=(dr != 0), use_noise=(gbs != 0))
                        update_status(ae, "current", "training encdec "+name)
                        h = ae.pretrain(name=name, early_stopping={"monitor":"acc", "patience":6, "verbose":1})
                        update_status(ae, "history", {"name": name,
                                                      "loss": losscomb(0, h),
                                                      "acc": losscomb(1, h)})
    ### FINETUNE ###
    if finetune:
        if not pretrain_encdecs:
            ae.load_encdecs("best")
        ### CONFIG ###
        ae_conf = dict(epochs=120
                      ,batch_size=60
                      ,drop_rate=0
                      ,gauss_base_sigma=0.0
                      ,gauss_sigma_factor=2
                      ,class_optimizer='adadelta'
                      ,class_loss=biased_cce)
        ### RUN ###
        #sink = Sink([ae], lambda r: print(r) if (not r is None) else None)
        #print('have sink')
        for opt in optimizers:
            for cl in class_losses:
                for dr in drop_rates:
                    for gbs in gauss_base_sigmas:
                        for gsf in gauss_sigma_factors:
                            ae.cls_opt = opt
                            ae.cls_lss = cl
                            ae.drop_rate = dr
                            ae.sigma_base = gbs
                            ae.sigma_fact = gsf
                            name = "o-"+str(opt)+"-cl-"+str(cl)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)
                            history = ae.finetune(train_encdecs=False)
                            print('finetuned')
                            for key in ae.previous_data:
                                print(key)
                                arr = np.array(ae.previous_data[key])
                                phl = len(phase_names)
                                eye = np.identity(phl)
                                eval= ae.model.evaluate(arr, np_utils.to_categorical(map(lambda n: phase_names.index(n)
                                                                                        ,list(repeat(key, len(arr))))
                                                                                    ,phl)
                                                       ,show_accuracy=True)
                                counts = np.sum(map(lambda p: eye[np.argmax(p)], ae.model.predict(arr)), axis=0)
                                print(eval)
                                print(counts)