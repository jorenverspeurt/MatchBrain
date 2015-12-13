from __future__ import print_function

import cPickle as pickle
import datetime as dt
import gzip
import json
import logging
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
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from core.TrainView import phase_names
from ml.logutils import LogSourceMaker
from ml.processing import Preprocessing
from signals.primitive import Transformer, Source

logger = logging.getLogger('learning')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

__author__ = 'joren'

time_str = lambda: dt.datetime.now().strftime('%y%m%d-%H%M%S')

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

        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            if self.val:
                self.losses.append(((logs.get('loss')
                                    ,logs.get('acc'))
                                   ,(logs.get('val_loss')
                                    ,logs.get('val_acc'))))
            else:
                self.losses.append((logs.get('loss'), logs.get('acc')))

    def __init__(self
                ,input_dim
                ,bw_source
                ,ph_source = Source(lambda: None)
                ,mode = TRAINING
                ,batch_size = 60
                ,epochs = 60
                ,num_sizes = 4
                ,encdec_optimizer = 'rmsprop'
                ,class_optimizer = 'adadelta'
                ,class_loss = 'categorical_crossentropy'
                ,drop_rate = 0.1
                ,gauss_base_sigma = 0.0
                ,gauss_sigma_factor = 2
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
        f = open(self.model_dir + self.catalog_name, 'r+')
        catalog = json.loads(f.read())
        if name in catalog:
            entry = catalog[name]
            allkeys = set(entry.keys()).update(info.key())
            catalog[name] = {key: info[key] if key in info else entry[key] for key in allkeys}
        else:
            catalog[name] = info
        f.write(json.dumps(catalog, indent=2, sort_keys=True))
        f.close()

    def training_transform(self, bw, ph):
        self.tuning_transform(bw, ph)
        #if self.batched == self.batch_size:
        #    X_l = np.array(map(np.array, self.current_batch))
        #    for (lay, ae) in enumerate(self.enc_decs):
        #        loss = ae.train_on_batch(X_l, X_l)
        #        logger.info({"training_loss_pre_"+str(lay): {self.getName(): loss}})
        #        X_l = ae.predict(X_l, batch_size=self.batch_size, verbose=0)
        #    self.batched = 0
        #    return X_l
        #else:
        #    # Is this necessary or even useful?
        #    x_l = np.array([bw])
        #    for ae in self.enc_decs:
        #        x_l = ae.predict(x_l, batch_size=1, verbose=0)
        #    return x_l

    def tuning_transform(self, bw, ph):
        self.current_phase = ph
        #self.maxes = np.maximum(self.maxes, np.abs(bw))
        scaled = np.divide(bw, self.maxes)
        self.current_batch = self.current_batch[1:]+[scaled]
        self.batched += 1
        self.previous_data.setdefault(self.current_phase, []).append(scaled)
        return None

    def using_transform(self, bw):
        return self.model.predict(np.array(bw), batch_size=1)

    def pull(self):
        return self.value

    def makeValue(self):
        new_val = self.transform(**self.inputs)
        if not new_val is None:
            self.value = new_val
            if self.debug(): self.debug(self.getName()+": "+self.value)
            self.subscriptions = {s : False for s in self.subscriptions.keys()}

    def pretrain(self, name = None, overwrite_latest = True, early_stopping = None):
        X_l = np.array(sum(self.previous_data.values(), []))
        cum_history = []
        for (lay, ae) in enumerate(self.enc_decs):
            history = self.History()
            callbacks = [history]
            if early_stopping:
                callbacks.append(EarlyStopping(**early_stopping))
            ae.fit(X_l, X_l, batch_size=self.batch_size, nb_epoch=self.epochs,
                   show_accuracy=True, callbacks=callbacks)
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
    def finetune(self, name = None, train_encdecs = True, early_stopping = None):
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
            self.load_encdecs()
        if not self.model:
            self.new_model()
        measurements, phases = zip(*[(m,p) for p in self.previous_data for m in self.previous_data[p]])
        X_train, X_test, y_train, y_test = train_test_split(
            map(np.array, measurements),
            np_utils.to_categorical(map(
                lambda n: phase_names.index(n),
                phases)),
            test_size=0.1
        )
        history = self.History()
        callbacks = [ModelCheckpoint("at-"+start_time+"-{epoch}-{acc:.4f}.hdf5", save_best_only=True), history]
        if early_stopping:
            callbacks.append(EarlyStopping(**early_stopping))
        self.model.fit(np.array(X_train), y_train, batch_size=self.batch_size, nb_epoch=self.epochs,
                       show_accuracy=True, validation_data=(np.array(X_test), y_test),
                       callbacks=callbacks)
        score = self.model.evaluate(np.array(X_test), y_test, show_accuracy=True, verbose=0)
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

    def new_encdecs(self, compile = True, use_dropout = False, use_noise = False):
        self.enc_decs = []
        self.enc_use_drop = use_dropout
        self.enc_use_noise = use_noise
        for (i,(n_in, n_out)) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            ae = Sequential()
            enc_l = []
            if use_noise:
                enc_l.append(noise.GaussianNoise(self.sigma_base*(self.sigma_fact**-i)))
            enc_l.append(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid'))
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
                self.model.add(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid'))
                #TODO ?
        self.model.add(core.Dense(input_dim=self.layer_sizes[-1]
                                  ,output_dim=len(phase_names)
                                  ,activation='softmax'))
        if compile:
            self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def load_model(self, f_name = None):
        if not self.model and not f_name:
            self.new_model(compile=False)
        elif f_name:
            if not self.model():
                f_name_sizes = self.get_from_catalog("layer_sizes", f_name)
                latest_sizes = self.get_from_catalog("layer_sizes")
                self.layer_sizes = f_name_sizes or latest_sizes
                self.new_model(fresh = True, compile = False)
            self.model_name = f_name
        else:
            pass #?
        self.model.load_weights(self.model_dir + self.prefix + self.model_name)
        self.model.compile(loss=self.cls_lss, optimizer=self.cls_opt)

    def save_model(self, f_name = None):
        name = f_name or self.model_name
        if not self.model:
            logger.error({"error": {self.getName(): "Tried to save, but no model!"}})
        else:
            self.update_catalog(name, self.model_info())
            f = self.model_dir + self.prefix + name
            self.model.save_weights(f, overwrite=True)

    def save_encdecs(self, f_name = None):
        assert len(self.enc_decs) > 0
        name = f_name or self.encdecs_name or self.model_name
        self.update_catalog("ed_" + name, self.encdec_info())
        base = self.model_dir + self.prefix + name
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "_ed_" + str(i)
            ed.save_weights(f, overwrite=True)

    def load_encdecs(self, f_name = None):
        name = f_name or self.encdecs_name or self.model_name or "latest"
        if not self.enc_decs:
            self.layer_sizes = self.get_from_catalog("layer_sizes", "ed_" + name)
            self.new_encdecs(compile=False)
        base = self.model_dir + self.prefix + name
        for (i, ed) in enumerate(self.enc_decs):
            f = base + "_ed_" + str(i)
            ed.load_weights(f)
            ed.compile(loss='mse', optimizer=self.enc_opt)
            # TODO add things like optimizer chosen to catalog

    def load_data(self, f_name = None): #TODO load latest or named
        name = f_name or self.get_from_catalog("data")
        with gzip.open(self.model_dir + self.prefix + "data" + name + ".pkl.gz", 'rb') as f:
            self.previous_data = pickle.load(f)

    def save_data(self, f_name = None): #TODO copy to 'latest' or save the name in the catalog
        name = f_name or time_str()
        self.update_catalog(self.model_name, {"data": name})
        with gzip.open(self.model_dir + self.prefix + "data" + name + ".pkl.gz", 'wb') as f:
            pickle.dump(self.previous_data, f)

    def encdec_info(self):
        info = {
            "epochs": self.epochs,
            "layer_sizes": self.layer_sizes,
            "encdec_optimizer": self.enc_opt,
            "enc_use_drop": self.enc_use_drop,
            "drop_rate": self.drop_rate,
            "enc_use_noise": self.enc_use_noise,
            "gaussian_base_sigma": self.sigma_base,
            "gaussian_sigma_factor": self.sigma_fact
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
            "gaussian_sigma_factor": self.sigma_fact
        }
        return info



# TODO write tests for AutoTransformer
if __name__ == '__main__':
    ### CONFIG ###
    shift = 4
    label_normalization = [0.82967276, 1.69463687, 1.74141838, 0.3860981,  0.34817388]
    #label_normalization = [0.47643503, 0.97313597, 1.0,        0.22171473, 0.19993695]
    #label_normalization = [1.0, 2.0, 2.0, 0.0, 0.0]
    def biased_cce(y_true, y_pred):
        return K.mean(K.categorical_crossentropy(y_pred, label_normalization * y_true), axis=-1)
    nest = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    ae_conf = dict(epochs=120
                  ,batch_size=60
                  ,drop_rate=0
                  ,gauss_base_sigma=0.0
                  ,gauss_sigma_factor=2
                  ,class_optimizer='adadelta'
                  ,class_loss=biased_cce)
    generate_data = False
    data_mins = 35
    finetune = True
    ### RUN ###
    l = LogSourceMaker()
    b = l.get_block(shift = shift)
    bws = b.sources[0]
    phs = b.sources[1]
    prep = Preprocessing(bws)
    logger.info("prep output dim: "+str(prep.output_dim))
    ae = AutoTransformer(prep.output_dim, prep, phs, epochs=120, drop_rate=0, class_optimizer='adadelta', class_loss=biased_cce)
    print('have ae')
    #sink = Sink([ae], lambda r: print(r) if (not r is None) else None)
    #print('have sink')
    if generate_data:
        t = data_mins*4
        b.start()
        while b.started and t>0:
            time.sleep(15)
            print(ae.batched)
            t -= 1
        if b.started:
            b.stop()
        print('stopped')
        #ae.save_data()
    else:
        ae.load_data()
    if finetune:
        history = ae.finetune(train_encdecs=False)
        print('finetuned')
    else:
        ae.load_model()
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

