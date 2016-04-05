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
epochs = 100
shift = 1

#SPECIFIC
data_mins = 30
generate_data = False
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
    if not cross_validate:
        l = LogSourceMaker(logfolder="/root/MatchBrain/logs/")
        b = l.get_block(shift = shift)
        bws = b.sources[0]
        phs = b.sources[1]
        prep = Preprocessing(bws)
        print("prep output dim: "+str(prep.output_dim))
        AutoTransformer.model_dir = "/root/MatchBrain/models/"
        ae = AutoTransformer(prep.output_dim, prep, phs, epochs=epochs, num_sizes=5)
        print("have ae")
    ### DATA ###
    if generate_data:
        update_status(ae, "current", "preprocessing data")
        b.start()
        while b.started and data_mins>0:
            time.sleep(60)
            print(ae.batched)
            data_mins -= 1
        if b.started:
            b.stop()
        print("stopped")
        ae.cap_data()
        update_status(ae, "current", "saving data")
        ae.save_data()
        print("data saved")
    else:
        pass
        #ae.load_data()
        #ae.cap_data()
    ### PRETRAIN ###
    losscomb = lambda zoo, h: ", ".join(map(lambda (i,e): str(i)+":"+('%.4f'%e[-1][zoo]), enumerate(h)))
    if pretrain_encdecs:
        n = nest(1,0.9)
        counter = 0
        options = [('adadelta', 0.001, 0.001, 2)]
        ls = [0.001]
        secondary = [(0, l2) for l2 in ls]
        for (opt, dr, gbs, gsf) in options:
        #for opt in ['rmsprop']#, 'adadelta', nest(1,0.9)]:
        #    for dr in [0.001]#drop_rates[0:-1]:
        #        for gbs in [0.001]#gauss_base_sigmas[0:3]:
        #            for gsf in [1.5]#[1, 1.1, 2]:
            for (l1, l2) in secondary:
                counter += 1
                print(str(counter)+" of "+str(len(options)*len(secondary)))
                name = "o-"+str(opt)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)+"-l1-"+str(l1)+"-l2-"+str(l2)
                #if not ae.get_from_catalog("layer_sizes", "ed_"+name):
                ae.enc_opt = opt
                ae.drop_rate = dr
                ae.sigma_base = gbs
                ae.sigma_fact = gsf
                ae.l1 = l1
                ae.l2 = l2
                ae.new_encdecs(use_dropout=(dr != 0), use_noise=(gbs != 0))
                update_status(ae, "current", "training encdec "+name)
                h = ae.pretrain(name=name)
                info = {"acc_pre_"+str(i): float(e[-1][1]) for (i,e) in enumerate(h)}
                info.update({("loss_pre_"+str(i)): float(e[-1][0]) for (i,e) in enumerate(h)})
                ae.update_catalog("ed_"+name,info)
                update_status(ae, "history", {"name": name,
                                              "loss": losscomb(0, h),
                                              "acc": losscomb(1, h),
                                              "date": time_str()})
    if continue_encdecs:
        opt, dr, gbs, gsf, l2 = 'adadelta', 0.001, 0.001, 2, 0.001
        name = "o-"+str(opt)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)+"-l1-"+str(0)+"-l2-"+str(l2)
        ae.enc_opt = opt
        ae.drop_rate = dr
        ae.sigma_base = gbs
        ae.sigma_fact = gsf
        ae.l1 = 0
        ae.l2 = l2
        ae.epochs = 20
        ae.load_encdecs(name)
        h = ae.pretrain(name=name)
        info = {"acc_pre_"+str(i): float(e[-1][1]) for (i,e) in enumerate(h)}
        info.update({("loss_pre_"+str(i)): float(e[-1][0]) for (i,e) in enumerate(h)})
        ae.update_catalog("ed_"+name,info)
        update_status(ae, "history", {"name": name,
                                      "loss": losscomb(0, h),
                                      "acc": losscomb(1, h),
                                      "date": time_str()})
    ### FINETUNE ###
    if finetune:
        ae.load_encdecs("best")
        encdec_opts = {key: ae.get_from_catalog(key, "ed_best") for key in encdec_opt_strings}
        #sink = Sink([ae], lambda r: print(r) if (not r is None) else None)
        #print('have sink')
        options = [(opt,cl,dr,gbs,gsf,l2)
                   for opt in ['adadelta']
                   for cl in ['categorical_crossentropy']
                   for dr in [0.001]
                   for gbs in [0.001]
                   for gsf in [2]
                   for l2 in [0.001]]
        for (opt,cl,dr,gbs,gsf,l2) in options:
            ae.cls_opt = opt
            ae.cls_lss = cl
            ae.drop_rate = dr
            ae.sigma_base = gbs
            ae.sigma_fact = gsf
            name = "o-"+str(opt)+"-cl-"+str(cl)+"-dr-"+str(dr)+"-gbs-"+str(gbs)+"-gsf-"+str(gsf)+"-l2-"+str(l2)
            update_status(ae, "current", "training classif "+name)
            history = ae.finetune(train_encdecs=False, early_stopping={"monitor": "val_acc", "patience": 10, "verbose": 1})
            info = {"history": history, "loss_fine": history[-1][0], "acc_fine": history[-1][1]}
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
                update_status(ae, "history", {"name": name,
                                              "loss": history[-1][0],
                                              "acc": history[-1][1],
                                              "date": time_str()})
            ae.update_catalog(name, info)
    if continue_finetune:
        ae.load_encdecs("o-adadelta-dr-0.001-gbs-0.001-gsf-2-l1-0-l2-0.001")
        ae.load_model()#"151215-194404")
        #print(ae.model.to_json())
        ae.load_data()
        ae.cap_data()
        history = ae.finetune(train_encdecs=False)
        name = ae.model_name
        info = {"history": history, "loss_fine": history[-1][0], "acc_fine": history[-1][1]}
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
            update_status(ae, "history", {"name": name,
                                          "loss": history[-1][0],
                                          "acc": history[-1][1],
                                          "date": time_str()})
        ae.update_catalog(name, info)
    if evaluate_model:
        info = {}
        if not (finetune or continue_finetune):
            ae.load_model()
            if not generate_data:
                ae.load_data()
                ae.cap_data()
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
    if cross_validate:
        l = LogSourceMaker(clean_seconds=5, logfolder="/root/MatchBrain/logs/", cross_val=True)
        for _ in xrange(len(l.all_dict)+1):
            b = l.get_block(shift = 1)
            bws = b.sources[0]
            phs = b.sources[1]
            prep = Preprocessing(bws)
            print("prep output dim: "+str(prep.output_dim))
            AutoTransformer.model_dir = "/root/MatchBrain/models/"
            ae = AutoTransformer(prep.output_dim, prep, phs, epochs=50, num_sizes=5)
            print("have ae")
            #update_status(ae, "current", "preprocessing data")
            #b.start()
            #while b.started and data_mins>0:
            #    time.sleep(60)
            #    print(ae.batched)
            #    data_mins -= 1
            #if b.started:
            #    b.stop()
            #print("stopped")
            ae.load_data()
            ae.cap_data()
            #update_status(ae, "current", "saving data")
            #ae.save_data()
            #print("data saved")
            ae.finetune(name="x-"+l.val_dict.keys()[0]+"-val", train_encdecs=True, test_data=l.rpp_val)
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
            l.cross_val_next()

    #update_status(ae, "done", True)
