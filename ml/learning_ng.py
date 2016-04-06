from __future__ import print_function

import cPickle
import datetime as dt
import glob
import gzip
import json
import logging
import os
import random
from itertools import repeat
from math import log
from operator import mul

import keras.layers.containers as containers
import keras.layers.core as core
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import noise
from keras.models import Sequential
from keras.regularizers import WeightRegularizer
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

__author__ = 'joren'

logger = logging.getLogger('learning')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

phase_names = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']

time_str = lambda: dt.datetime.now().strftime('%y%m%d-%H%M%S')

class MyEarlyStopping(EarlyStopping):
    def __init__(self, monitor="acc", patience=10, verbose=0, desired="high"):
        EarlyStopping.__init__(self, monitor=monitor, patience=patience, verbose=verbose)
        self.logs = {}
        if desired == "high":
            self.check_improved = lambda new, old: new > old
            self.best = -self.best
        else:
            self.check_improved = lambda new, old: new < old

    # TODO change this so it averages per-batch statistics to per-epoch
    def on_epoch_end(self, epoch, logs={}):
        self.logs.update(logs)
        current = self.logs.get(self.monitor)
        if self.check_improved(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %03d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1

    # TODO change this so it keeps per-batch statistics in a list
    def on_batch_end(self, epoch, logs={}):
        self.logs.update(logs)


class History(Callback):
    def __init__(self, val = False):
        Callback.__init__(self)
        self.val = val
        self.losses = []
        self.batch_losses = []
        self.batch_accs = []

    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        l = logs or {}
        self.batch_losses.append(l.get('loss'))
        self.batch_accs.append(l.get('acc'))

    def on_epoch_end(self, epoch, logs=None):
        l = logs or {}
        epoch_loss = sum(self.batch_losses)/len(self.batch_losses)
        epoch_acc = sum(self.batch_accs)/len(self.batch_accs)
        if self.val:
            self.losses.append(((epoch_loss
                                ,epoch_acc)
                               ,(l.get('val_loss')
                                ,l.get('val_acc'))))
        else:
            self.losses.append((epoch_loss, epoch_acc))
        self.batch_losses = []
        self.batch_accs = []

def safe_head(l):
    return l[0] if l else None

class PretrainedClassifier(object):
    def __init__( self
                , data
                , labels
                , batch_size = 50
                , epochs = 300
                , model_name = ""
                , model_dir = ""
                , encdecs_name = ""
                , encdec_optimizer = 'adadelta'
                , class_optimizer = 'adadelta'
                , class_loss = 'categorical_crossentropy'
                , drop_rate = 0.0
                , gauss_base_sigma = 0.0
                , gauss_sigma_factor = 1.0
                , l1 = 0.0
                , l2 = 0.0 ):
        input_dim = len(safe_head(data))
        output_dim = len(set(labels))
        upper = int(log(input_dim)/log(2))
        lower = int(log(output_dim)/log(2))+1
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_sizes = [input_dim] + [2**i for i in xrange(upper, lower, -1)]
        self.enc_decs = []
        self.model = None
        self.model_dir = model_dir or safe_head(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'models')))+"/" or ""
        self._catalog_path = os.path.join(self.model_dir, 'ng_catalog.json')
        self.batch_size = batch_size
        self.enc_opt = encdec_optimizer
        self.cls_opt = class_optimizer
        self.cls_lss = class_loss
        self.drop_rate = drop_rate
        self.sigma_base = gauss_base_sigma
        self.sigma_fact = gauss_sigma_factor
        self.enc_use_drop = False
        self.enc_use_noise = False
        self.l1 = l1
        self.l2 = l2
        self.model_name = model_name or (""
                                        +"edo_"+self.enc_opt
                                        +"-co_"+self.cls_opt
                                        +"-cl_"+self.cls_lss
                                        +("-dr_"+str(self.drop_rate) if self.drop_rate > 0 else "")
                                        +("-sb_"+str(self.sigma_base)
                                         +"-sf_"+str(self.sigma_fact) if self.sigma_base > 0 else "")
                                        +("-l1_"+str(self.l1) if self.l1 > 0 else "")
                                        +("-l2_"+str(self.l2) if self.l2 > 0 else ""))
        self.encdecs_name = encdecs_name or "e-"+self.model_name
        self.model_name = "m-"+self.model_name if self.model_name else self.model_name
        self._model_info = {}
        self._encdec_info = {}
        # if encdecs_name:
        #     self.load_encdecs(encdecs_name)
        # elif model_name:
        #     self.load_model(model_name)

    def catalog_get(self, param_name, model_name = "latest"):
        if glob.glob(self._catalog_path):
            with open(self._catalog_path, 'r') as f:
                catalog = json.loads(f.read())
                if model_name in catalog and param_name in catalog[model_name]:
                    return catalog[model_name][param_name]
                else:
                    return None
        else:
            return None

    def catalog_set(self, value, param_name, model_name = "latest"):
        catalog = { model_name: { param_name: value } }
        if glob.glob(self._catalog_path):
            with open(self._catalog_path, 'r') as f:
                catalog = json.loads(f.read())
                if model_name in catalog:
                    catalog[model_name][param_name] = value
                else:
                    catalog[model_name] = { param_name: value }
        with open(self._catalog_path, 'w') as f:
            f.write(json.dumps(catalog, indent=2, sort_keys=True))


    def catalog_update(self, info, model_name):
        catalog = { model_name: info }
        if glob.glob(self._catalog_path):
            with open(self._catalog_path, 'r') as f:
                catalog = json.loads(f.read())
                if model_name in catalog:
                    entry = catalog[model_name]
                    allkeys = set(entry.keys())
                    allkeys.update(info.keys())
                    catalog[model_name] = {key: (info[key] if key in info else entry[key]) for key in allkeys}
                else:
                    catalog[model_name] = info
        with open(self._catalog_path, 'w') as f:
            f.write(json.dumps(catalog, indent=2, sort_keys=True))

    def pretrain(self, name = None, overwrite_latest = True, overwrite_best = True, early_stopping = None):
        if not self.enc_decs:
            self.new_encdecs()
        f_name = name or time_str()
        self.encdecs_name = self.encdecs_name or f_name
        X_l = np.array(self.data)
        cum_history = []
        for (lay, ae) in enumerate(self.enc_decs):
            history = History()
            callbacks = [history]
            if early_stopping:
                callbacks.append(MyEarlyStopping(**early_stopping))
            ae.layers[0].output_reconstruction = True
            ae.compile(loss='mse', optimizer=self.enc_opt)
            ae.fit(X_l, X_l, batch_size=self.batch_size, nb_epoch=self.epochs // (2 ** lay),
                   show_accuracy=True, callbacks=callbacks, verbose=2)
            ae.layers[0].output_reconstruction = False
            ae.compile(loss='mse', optimizer=self.enc_opt)
            X_l = ae.predict(X_l, batch_size=self.batch_size, verbose=0)
            cum_history.append(history.losses)
        quality = reduce(mul, (i[-1][1] for i in cum_history))
        self._encdec_info['train_quality'] = quality
        self.evaluate_encdecs()
        self.save_encdecs(self.encdecs_name)
        if overwrite_latest:
            self.save_encdecs('latest')
        if quality > (self.catalog_get('quality', 'e-best') or 0) and overwrite_best:
            self.save_encdecs('best')
        return cum_history

    def finetune(self,
                 name = None,
                 encdecs_name = "",
                 early_stopping = {"monitor": "val_acc", "patience": 1000, "verbose": 1},
                 test_data = None,
                 overwrite_best = True):
        """
        :param train_encdecs: If true, pretraining is done now, if not the latest pretrained layers are loaded
        :param early_stopping: Should be None or a dict with keys "monitor", "patience", and possibly "verbose"
        :return: The loss and accuracy history of the model fit, type [((loss,acc),(val_loss,val_acc))]
        """
        start_time = time_str()
        if encdecs_name:
            self.load_encdecs(encdecs_name)
        if not self.model:
            self.new_model()
        X_train, X_val, y_train, y_val = train_test_split(
            # map(np.array, self.data), # shouldn't be necessary
            self.data,
            np_utils.to_categorical(map(
                lambda n: phase_names.index(n) if n else phase_names.index("DISTRACT"),
                self.labels)),
            test_size=0.1
        )
        if not test_data:
            X_test, y_test = np.array(X_val), y_val
        else:
            ph_t, m_t = zip(*test_data)
            X_test = np.array(m_t)
            y_test = np_utils.to_categorical(map(
                lambda n: phase_names.index(n) if n else phase_names.index("DISTRACT"),
            ph_t))
        history = History()
        callbacks = [ModelCheckpoint("m-"+start_time+"-{epoch}-{val_acc:.4f}.hdf5", save_best_only=True), history]
        if early_stopping:
            callbacks.append(MyEarlyStopping(**early_stopping))
        self.model.fit(np.array(X_train), y_train, batch_size=self.batch_size, nb_epoch=self.epochs,
                       show_accuracy=True, validation_data=(np.array(X_val), y_val),
                       callbacks=callbacks)
        score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
        logger.info({"finetune_score": score})
        self._model_info['test_loss'] = score[0]
        self._model_info['test_accuracy'] = score[1]
        self.evaluate_model()
        self.save_model(name or self.model_name or time_str())
        self.save_model('latest')
        if overwrite_best and score[1] > (self.catalog_get('test_accuracy', 'best') or 0):
            self.save_model('best')
        return history.losses

    def new_encdecs(self, compile = True, use_dropout = None, use_noise = None):
        self.enc_decs = []
        if not use_dropout is None:
            self.enc_use_drop = use_dropout
        if not use_noise is None:
            self.enc_use_noise = use_noise
        if self.l1 != 0 or self.l2 != 0:
            regularizer = WeightRegularizer(l1=self.l1, l2=self.l2)
        else:
            regularizer = None
        for (i,(n_in, n_out)) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            ae = Sequential()
            enc_l = []
            if self.enc_use_noise:
                enc_l.append(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i), input_shape=(n_in,)))
            enc_l.append(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid', W_regularizer=regularizer))
            if self.enc_use_drop:
                enc_l.append(core.Dropout(self.drop_rate))
            enc = containers.Sequential(enc_l)
            dec = containers.Sequential([core.Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')])
            ae.add(core.AutoEncoder(encoder=enc, decoder=dec,
                                    output_reconstruction=True))
            if compile:
                ae.compile(loss='mse', optimizer=self.enc_opt)
            self.enc_decs.append(ae)

    def new_model(self, fresh = False, compile = True):
        self.model = Sequential()
        drop_cl = core.Dropout
        if self.l1 != 0 or self.l2 != 0:
            regularizer = WeightRegularizer(l1=self.l1, l2=self.l2)
        else:
            regularizer = None
        if self.enc_decs and not fresh:
            # The encoder may already have a noise and/or drop layer!
            # But we don't know what kind so replace them
            for (i,enc) in enumerate(ae.layers[0].encoder.layers[0 if len(ae.layers[0].encoder.layers) == 0 else 1] for ae in self.enc_decs):
                if self.sigma_base != 0:
                    self.model.add(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i), input_shape=(self.layer_sizes[i], )))
                self.model.add(enc)
                if self.drop_rate != 0:
                    self.model.add(drop_cl(self.drop_rate))
        else:
            for (i,(n_in, n_out)) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
                if self.sigma_base != 0:
                    self.model.add(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i), input_shape=(self.layer_sizes[i], )))
                self.model.add(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid', W_regularizer=regularizer))
                if self.drop_rate != 0:
                    self.model.add(drop_cl(self.drop_rate, input_shape=(n_in,)))
                #TODO ?
        self.model.add(core.Dense(input_dim=self.layer_sizes[-1]
                                  ,output_dim=len(phase_names)
                                  ,activation='softmax'
                                  ,W_regularizer=regularizer))
        if compile:
            self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def load_model(self, f_name = None):
        self.model_name = f_name or self.model_name
        self.layer_sizes = self.catalog_get("layer_sizes", self.model_name)
        self.cls_opt = self.catalog_get("class_optimizer", self.model_name)
        self.cls_lss = self.catalog_get("class_loss", self.model_name)
        self.sigma_base = self.catalog_get("gaussian_base_sigma", self.model_name)
        self.sigma_fact = self.catalog_get("gaussian_sigma_factor", self.model_name)
        self.new_model(fresh=True, compile=False)
        self.model.load_weights(self.model_dir + self.model_name)
        self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def save_model(self, f_name = None):
        name = f_name or self.model_name
        if not self.model:
            logger.error({"error": "Tried to save, but no model!"})
        else:
            f = self.model_dir + name
            self.model.save_weights(f, overwrite=True)
            self.catalog_update(self.model_info(), name)

    def save_encdecs(self, f_name = None):
        assert len(self.enc_decs) > 0
        name = "e-" + f_name or self.encdecs_name or "e-" + self.model_name
        base = self.model_dir + name
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "-" + str(i)
            ed.save_weights(f, overwrite=True)
        self.catalog_update(self.encdec_info(), name)

    def load_encdecs(self, f_name = None):
        name = "e-" + f_name or self.encdecs_name or "e-" + self.model_name or "e-latest"
        get_cat = lambda item: self.catalog_get(item, name)
        if not self.enc_decs:
            self.layer_sizes = get_cat("layer_sizes")
            use_drop = get_cat("enc_use_drop")
            use_noise = get_cat("enc_use_noise")
            self.new_encdecs(compile=False, use_dropout=use_drop, use_noise=use_noise)
        base = self.model_dir + name
        self.enc_opt = get_cat("encdec_optimizer")
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "-" + str(i)
            ed.load_weights(f)
            ed.compile(loss='mse', optimizer=self.enc_opt)

    def cap_data(self):
        data_by_key = {}
        for (d, l) in zip(self.data, self.labels):
            data_by_key.setdefault(l, []).append(d)
        lengths = map(len, data_by_key.values())
        if not all(map(lambda a: a == lengths[0], lengths)):
            smallest = min(lengths)
            for key in data_by_key:
                data_by_key[key] = random.sample(data_by_key[key], smallest)
            self.data, self.labels = zip(*[(d,l)
                                           for (l,ds) in data_by_key.iteritems()
                                           for d in ds])

    def evaluate_model(self, label_data_pairs = None):
        if self.model:
            pairs = label_data_pairs or zip(self.labels, self.data)
            data_by_label = {}
            for (l, d) in pairs:
                data_by_label.setdefault(l, []).append(d)
            info = {}
            phl = len(phase_names)
            eye = np.identity(phl)
            tc  = np_utils.to_categorical
            for key in data_by_label:
                arr = np.array(data_by_label[key])
                eva = self.model.evaluate(arr
                                         ,tc(map(lambda n: phase_names.index(n)
                                                ,list(repeat(key, len(arr))))
                                            ,phl)
                                         ,batch_size=self.batch_size
                                         ,show_accuracy=True)
                counts = np.sum(map(lambda p: eye[np.argmax(p)]
                                   ,self.model.predict(arr, batch_size=self.batch_size))
                               ,axis=0)
                info[key] = { 'loss': float(eva[0])
                            , 'accuracy': float(eva[1])
                            , 'counts': map(int, list(counts)) }
            self._model_info.update(info)

    def evaluate_encdecs(self, data = None):
        if self.enc_decs:
            data = data or np.array(self.data)
            info = { 'quality': 1 }
            for (i,ed) in enumerate(self.enc_decs):
                ed.layers[0].output_reconstruction = True
                ed.compile(loss='mse', optimizer=self.enc_opt)
                eva = ed.evaluate(data, data, show_accuracy=True, verbose=0, batch_size= self.batch_size)
                info["ed-"+str(i)] = { 'loss': float(eva[0])
                                     , 'accuracy': float(eva[1]) }
                info['quality'] *= float(eva[1])
                ed.layers[0].output_reconstruction = False
                ed.compile(loss='mse', optimizer=self.enc_opt)
                data = ed.predict(data, verbose=0, batch_size=self.batch_size)
            self._encdec_info.update(info)


    def encdec_info(self, evaluate = None, eval_data = None):
        if evaluate or (evaluate is None and self.enc_decs):
            self.evaluate_encdecs(eval_data)
        info = {
            "epochs": self.epochs,
            "layer_sizes": self.layer_sizes,
            "encdec_optimizer": self.enc_opt,
            "enc_use_drop": self.enc_use_drop,
            "drop_rate": self.drop_rate,
            "enc_use_noise": self.enc_use_noise,
            "gaussian_base_sigma": self.sigma_base,
            "gaussian_sigma_factor": self.sigma_fact,
            "l1": self.l1,
            "l2": self.l2
        }
        info.update(self._encdec_info)
        return info

    def model_info(self, evaluate=None):
        if evaluate or (evaluate is None and self.model):
            self.evaluate_model()
        info = {
            "epochs": self.epochs,
            "layer_sizes": self.layer_sizes,
            "class_optimizer": self.cls_opt,
            "class_loss": self.cls_lss,
            "drop_rate": self.drop_rate,
            "gaussian_base_sigma": self.sigma_base,
            "gaussian_sigma_factor": self.sigma_fact,
            "l1": self.l1,
            "l2": self.l2
        }
        info.update(self._model_info)
        return info


def default_instance(file_name = None):
    location = file_name or safe_head(glob.glob(os.path.join(os.path.dirname(__file__), 'normalized.pkl.gz')))
    with gzip.open(location, 'rb') as f:
        normalized_data = cPickle.load(f)
    unsplit = [(e['phase'],e['raw']) for name in normalized_data.iterkeys() for e in normalized_data[name]]
    phases, data = zip(*unsplit)
    return PretrainedClassifier(data, phases, 50, 100, model_name = 'test-'+time_str(), gauss_base_sigma=0.1, gauss_sigma_factor=2, l2=0.001)


if __name__ == '__main__':
    pc = default_instance()
    print(pc.layer_sizes)
    pc.new_encdecs(True,True,True)
    pc.pretrain()
    pc.cap_data()
    pc.finetune()
