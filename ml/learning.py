from __future__ import print_function

import cPickle as pickle
import datetime as dt
import gzip
import json
import logging
import random
import time
from itertools import repeat
from operator import mul

import keras.layers.containers as containers
import keras.layers.core as core
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import noise
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import WeightRegularizer
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from ml.logutils import LogSourceMaker
from ml.processing import Preprocessing
from signals.primitive import Transformer, Source, Sink

logger = logging.getLogger('learning')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

phase_names = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']

__author__ = 'joren'

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


class AutoTransformer(Transformer):
    """
    An autoencoder, forced to be a signals.primitive.Transformer.
    The layers of the autoencoder are determined by the layer_sizes attribute. The layers are pretrained while the
    model is in training mode
    """
    model_dir = '/home/joren/PycharmProjects/MatchBrain/models/'
    prefix = 'at-'
    catalog_name = 'catalog.json'

    class Mode(object):
        def __init__(self, name, num_sources, t_name, t_assignments, no_updates):
            """
            :param t_name: Name for the transform method to be used
            :param t_assignments: Either a string->string dict (i.e. {"a1b2->": "bw"}) or a formattable string
            :param no_updates: An iterable with values from t_assignments.values().
            :return: A non-concrete AutoTransformer.Mode. Made concrete by calling instantiate.
            """
            self.name = name
            self.num_sources = num_sources
            self.sources = None
            self.t_name = t_name
            self.transform = None
            self.tass_string = str(t_assignments)
            self.t_assignments = None
            self.no_updates = no_updates

        def instantiate(self, parent, sources):
            result = AutoTransformer.Mode(self.name
                                         ,self.num_sources
                                         ,self.t_name
                                         ,self.t_assignments
                                         ,self.no_updates)
            result.sources = sources[0:self.num_sources]
            result.transform = getattr(parent, self.t_name)
            result.t_assignments = eval(self.tass_string.format(*map(lambda s: s.getName(), result.sources)))
            return result

    TRAINING = Mode('training'
                   ,2
                   ,'training_transform'
                   ,'{{ "{}": "bw", "{}": "ph" }}'
                   ,'[ph]')
    TUNING   = Mode('tuning'
                   ,2
                   ,'tuning_transform'
                   ,'{{ }}'
                   ,'[bw, ph]')
    USING    = Mode('using'
                   ,1
                   ,'using_transform'
                   ,'{{ "{}": "bw" }}'
                   ,'[]')

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


    def __init__(self
                ,input_dim
                ,bw_source
                ,ph_source = Source(lambda: None)
                ,mode = TRAINING
                ,batch_size = 60
                ,epochs = 60
                ,num_sizes = 5
                ,encdec_optimizer = 'rmsprop'
                ,class_optimizer = 'adadelta'
                ,class_loss = 'categorical_crossentropy'
                ,drop_rate = 0.001
                ,gauss_base_sigma = 0.001
                ,gauss_sigma_factor = 2
                ,l1 = 0.0
                ,l2 = 0.001
                ,model_name = 'latest'
                ,encdecs_name = None):
        if mode is AutoTransformer.TUNING: raise ValueError("Can't instantiate an AT in 'tuning' mode")
        self.mode = mode.instantiate(self, [bw_source, ph_source]) # Source order matters!
        self.batch_size = batch_size
        self.epochs = epochs
        self.bw_source = bw_source
        self.ph_source = ph_source
        self.layer_sizes = [input_dim] + [2**i for i in xrange(6, 6-num_sizes+1, -1)]
        self.enc_decs = []
        self.current_batch = [[] for i in range(batch_size)]
        self.previous_data = {}
        self.current_phase = phase_names[0]
        self.batched = 0
        self.model = None
        self.model_name = model_name
        self.encdecs_name = encdecs_name
        self.maxes = self.get_from_catalog("maxes", model_name) or np.ones(input_dim)
        self.enc_opt = encdec_optimizer
        self.cls_opt = class_optimizer
        self.cls_lss = class_loss
        self.drop_rate = drop_rate
        self.sigma_base = gauss_base_sigma
        self.sigma_fact = gauss_sigma_factor
        self.best_encdecs = ("latest",0.0)
        self.enc_use_drop = False
        self.enc_noise = False
        self.l1 = l1
        self.l2 = l2
        Transformer.__init__(self
                            ,self.mode.sources
                            ,self.mode.t_assignments
                            ,self.mode.transform
                            ,no_updates=self.mode.no_updates)
        if self.mode.name == 'training':
            self.new_encdecs()
        else:
            self.load_model()

    def get_from_catalog(self, param_name, model_name = "latest"):
        with open(self.model_dir + self.catalog_name) as f:
            catalog = json.loads(f.read())
            if model_name in catalog and param_name in catalog[model_name]:
                return catalog[model_name][param_name]
            else:
                return None

    def update_catalog(self, name, info):
        with open(self.model_dir + self.catalog_name, 'r') as f:
            catalog = json.loads(f.read())
            if name in catalog:
                entry = catalog[name]
                allkeys = set(entry.keys())
                allkeys.update(info.keys())
                catalog[name] = {key: (info[key] if key in info else entry[key]) for key in allkeys}
            else:
                catalog[name] = info
        with open(self.model_dir + self.catalog_name, 'w') as f:
            f.write(json.dumps(catalog, indent=2, sort_keys=True))

    def training_transform(self, bw, ph, online=False):
        self.tuning_transform(bw, ph)
        if online:
            X_l = np.array([self.current_batch[-1]])
            for (lay, ae) in enumerate(self.enc_decs):
                loss = ae.train_on_batch(X_l, X_l)
                X_l = ae.predict(X_l, batch_size=1, verbose=0)
            return X_l
        else:
            return None
        #if self.batched == self.batch_size:
        #    X_l = np.array(map(np.array, self.current_batch))
        #    for (lay, ae) in enumerate(self.enc_decs):
        #        loss = ae.train_on_batch(X_l, X_l)
        #        logger.info({"training_loss_pre_"+str(lay): {self.getName(): loss}})
        #        X_l = ae.predict(X_l, batch_size=self.batch_size, verbose=0)
        #    self.batched = 0
        #    return X_l
        # else:
        #     return None

    def tuning_transform(self, bw, ph):
        self.current_phase = ph
        #self.maxes = np.maximum(self.maxes, np.abs(bw))
        if ph and (not bw is None) and len(bw)>0:
            #scaled = np.divide(bw, self.maxes)
            self.current_batch = self.current_batch[1:]+[bw]
            self.batched += 1
            self.previous_data.setdefault(self.current_phase, []).append(bw)
        return None

    def using_transform(self, bw):
        return self.model.predict(np.array([bw]), batch_size=1)

    def pull(self):
        return self.value

    def push(self, source, value = None):
        if source in self.subscriptions:
            if not value is None:
                self.subscriptions[source] = True
                input_name = self.getInputNameFor(source)
                if input_name:
                    self.inputs[input_name] = value
                    self.makeValue()
                    for scriber in self.subscribers:
                        scriber.push(self, self.value)
            else:
                self.subscriptions[source] = False

    def makeValue(self):
        inputs = {key: self.inputs[key] for source in self.subscriptions for key in (self.getInputNameFor(source),)}
        new_val = self.transform(**inputs)
        if not new_val is None:
            self.value = new_val
            if self.debug(): self.debug(self.getName()+": "+self.value)
            self.subscriptions = {s : False for s in self.subscriptions.keys()}

    def pretrain(self, name = None, overwrite_latest = True, early_stopping = None):
        X_l = np.array(sum(self.previous_data.values(), []))
        print(len(X_l))
        cum_history = []
        for (lay, ae) in enumerate(self.enc_decs):
            history = self.History()
            callbacks = [history]
            if early_stopping:
                callbacks.append(MyEarlyStopping(**early_stopping))
            ae.fit(X_l, X_l, batch_size=self.batch_size, nb_epoch=self.epochs//(lay+1),
                   show_accuracy=True, callbacks=callbacks, verbose=2)
            X_l = ae.predict(X_l, batch_size=self.batch_size, verbose=0)
            cum_history.append(history.losses)
        f_name = name or time_str()
        self.save_encdecs(f_name)
        self.encdecs_name = f_name
        if overwrite_latest:
            self.save_encdecs()
        quality = reduce(mul, (i[-1][1] for i in cum_history))
        if quality > self.best_encdecs[1]:
            self.best_encdecs = (f_name, quality)
            self.save_encdecs("best")
        return cum_history

    # TODO make this run in a separate thread? See to it that all incoming signals are handled correctly during
    # TODO make this accept validation data from elsewhere or at least have x-validation option
    def finetune(self, name = None, train_encdecs = True, early_stopping = None, test_data = None):
        """
        :param train_encdecs: If true, pretraining is done now, if not the latest pretrained layers are loaded
        :param early_stopping: Should be None or a dict with keys "monitor", "patience", and possibly "verbose"
        :return: The loss and accuracy history of the model fit, type [((loss,acc),(val_loss,val_acc))]
        """
        self.change_mode(self.TUNING)
        start_time = time_str()
        if train_encdecs:
            self.pretrain()
        else:
            pass
            #self.load_encdecs()
        if not self.model:
            self.new_model()
        measurements, phases = zip(*[(m,p) for p in self.previous_data for m in self.previous_data[p]])
        X_train, X_val, y_train, y_val = train_test_split(
            map(np.array, measurements),
            np_utils.to_categorical(map(
                lambda n: phase_names.index(n) if n in phase_names else phase_names.index("DISTRACT"),
                phases)),
            test_size=0.1
        )
        if not test_data:
            X_test, y_test = np.array(X_val), y_val
        else:
            print(len(test_data))
            print(test_data[0])
            ph_t, m_t = zip(*test_data)
            X_test = np.array(map(lambda m: np.divide(m,self.maxes), m_t))
            y_test = np_utils.to_categorical(map(
                lambda n: phase_names.index(n) if n in phase_names else phase_names.index("DISTRACT"),
            ph_t))
        history = self.History()
        callbacks = [ModelCheckpoint("at-"+start_time+"-{epoch}-{val_acc:.4f}.hdf5", save_best_only=True), history]
        if early_stopping:
            callbacks.append(MyEarlyStopping(**early_stopping))
        self.model.fit(np.array(X_train), y_train, batch_size=self.batch_size, nb_epoch=self.epochs,
                       show_accuracy=True, validation_data=(X_val, y_val),
                       callbacks=callbacks)
        score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
        logger.info({"finetune_score": score})
        f_name = name or time_str()
        self.save_model(f_name)
        self.save_model()
        #self.change_mode(self.USING)
        return history.losses

    def change_mode(self, new_mode):
        self.mode = new_mode.instantiate(self, [self.bw_source, self.ph_source])
        self.transform = self.mode.transform
        self.arg_assigns = self.mode.t_assignments
        self.no_updates = self.mode.no_updates
        # DIRTY
        self.setSources(self.mode.sources)
        if new_mode.name == self.USING.name:
            Sink([self], lambda d: logging.getLogger('data.at').info(list(d)))

    def new_encdecs(self, compile = True, use_dropout = False, use_noise = False):
        self.enc_decs = []
        self.enc_use_drop = use_dropout
        self.enc_use_noise = use_noise
        if self.l1 != 0 or self.l2 != 0:
            regularizer = WeightRegularizer(l1=self.l1, l2=self.l2)
        else:
            regularizer = None
        for (i,(n_in, n_out)) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            ae = Sequential()
            enc_l = []
            if use_noise:
                enc_l.append(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i), input_shape=(n_in,)))
            enc_l.append(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid', W_regularizer=regularizer))
            if use_dropout:
                enc_l.append(core.Dropout(self.drop_rate))
            enc = containers.Sequential(enc_l)
            dec = containers.Sequential([core.Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')])
            ae.add(core.AutoEncoder(encoder=enc, decoder=dec,
                                    output_reconstruction=False))
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
            for (i,enc) in enumerate(ae.layers[0].encoder for ae in self.enc_decs):
                if self.drop_rate != 0:
                    self.model.add(drop_cl(self.drop_rate, input_shape=(self.layer_sizes[i],)))
                if self.sigma_base != 0:
                    self.model.add(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i)))
                self.model.add(enc)
        else:
            for (i,(n_in, n_out)) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
                if self.drop_rate != 0:
                    self.model.add(drop_cl(self.drop_rate, input_shape=(n_in,)))
                if self.sigma_base != 0:
                    self.model.add(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i)))
                self.model.add(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid', W_regularizer=regularizer))
                #TODO ?
        self.model.add(core.Dense(input_dim=self.layer_sizes[-1]
                                  ,output_dim=len(phase_names)
                                  ,activation='softmax'
                                  ,W_regularizer=regularizer))
        if compile:
            self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def load_model(self, f_name = None):
        self.model_name = f_name or self.model_name
        if not self.get_from_catalog("layer_sizes", self.model_name):
            raise ValueError("Wrong model name?")
        self.layer_sizes = self.get_from_catalog("layer_sizes", self.model_name)
        self.cls_opt = self.get_from_catalog("class_optimizer", self.model_name)
        self.cls_lss = self.get_from_catalog("class_loss", self.model_name)
        self.sigma_base = self.get_from_catalog("gaussian_base_sigma", self.model_name)
        self.sigma_fact = self.get_from_catalog("gaussian_sigma_factor", self.model_name)
        self.new_model(fresh=True, compile=False)
        self.model.load_weights(self.model_dir + self.prefix + self.model_name)
        self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def save_model(self, f_name = None):
        name = f_name or self.model_name
        if not self.model:
            logger.error({"error": {self.getName(): "Tried to save, but no model!"}})
        else:
            f = self.model_dir + self.prefix + name
            self.model.save_weights(f, overwrite=True)
            self.update_catalog(name, self.model_info())

    def save_encdecs(self, f_name = None):
        assert len(self.enc_decs) > 0
        name = f_name or self.encdecs_name or self.model_name
        base = self.model_dir + self.prefix + name
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "_ed_" + str(i)
            ed.save_weights(f, overwrite=True)
        self.update_catalog("ed_" + name, self.encdec_info())

    def load_encdecs(self, f_name = None):
        name = f_name or self.encdecs_name or self.model_name or "latest"
        get_cat = lambda item: self.get_from_catalog(item, "ed_"+name)
        if not self.enc_decs:
            self.layer_sizes = get_cat("layer_sizes")
            use_drop = get_cat("enc_use_drop")
            use_noise = get_cat("enc_use_noise")
            self.new_encdecs(compile=False, use_dropout=use_drop, use_noise=use_noise)
        base = self.model_dir + self.prefix + name
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "_ed_" + str(i)
            ed.load_weights(f)
            self.enc_opt = get_cat("encdec_optimizer")
            ed.compile(loss='mse', optimizer=self.enc_opt)

    def cap_data(self):
        self.previous_data = {k:self.previous_data[k] for k in phase_names}
        lengths = map(len, self.previous_data.values())
        if not all(map(lambda a: a == lengths[0], lengths)):
            smallest = min(map(len, self.previous_data.values()))
            for key in self.previous_data:
                self.previous_data[key] = random.sample(self.previous_data[key], smallest)

    def load_data(self, f_name = None):
        name = f_name or self.get_from_catalog("data")
        with gzip.open(self.model_dir + self.prefix + "data" + name + ".pkl.gz", 'rb') as f:
            self.previous_data = pickle.load(f)

    def save_data(self, f_name = None):
        name = f_name or time_str()
        with gzip.open(self.model_dir + self.prefix + "data" + name + ".pkl.gz", 'wb') as f:
            pickle.dump(self.previous_data, f)
        self.update_catalog(self.model_name, {"data": name})

    def encdec_info(self):
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
        return info

    def model_info(self):
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
        return info


# TODO write tests for AutoTransformer
