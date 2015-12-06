from __future__ import print_function

import json
import logging
import time

import keras.layers.containers as containers
import keras.layers.core as core
import keras.layers.noise as noise
import numpy as np
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from core.TrainView import phase_names
from ml.logutils import LogSourceMaker
from ml.processing import Preprocessing
from signals.primitive import Transformer, Accumulator, Source

logger = logging.getLogger('learning')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

__author__ = 'joren'

class Autoencoder(Transformer):
    """
    A first attempt at writing an AutoEncoder Transformer with Keras...
    """
    def __init__(self, source, input_dim, train):
        self.batch_size = 15
        if train:
            self.model = Sequential()
            #T pick good sigma
            encoder = containers.Sequential(layers=[
                core.Dropout(0.5, input_shape=(input_dim,))
            ])
            dense_opts = {
                #T magic value
                'activation': 'sigmoid',
                #T magic number
                'W_regularizer': l2(0.01)
            }
            enc_others = [
                core.Dense(
                    input_dim=input_dim//(2**i),
                    output_dim=input_dim//(2**(i+1)),
                    **dense_opts
                )
                #T magic number(s)
                for i in xrange(5) if input_dim//(2**(i+1)) > 8
            ]
            enc_others.append(core.Dense(input_dim=enc_others[-1].output_dim, output_dim=8, **dense_opts))
            for dense in enc_others:
                encoder.add(dense)
                #T magic
                encoder.add(noise.GaussianNoise(1))
            dec_layers = [
                core.Dense(
                    input_dim=dense.output_dim,
                    output_dim=dense.input_dim,
                    **dense_opts
                )
                for dense in enc_others[::-1]
            ]
            decoder = containers.Sequential(layers=dec_layers)
            self.model.add(core.AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
            self.model.compile(optimizer='adagrad', loss='mse')
            logger.debug("compiled model")
            self.raw_source = source
            accumulator = Accumulator(source, self.batch_size, [0 for i in xrange(self.batch_size)])
            logger.debug("raw_source: "+source.getName())
            logger.debug("accumulator: "+accumulator.getName())
            logger.debug("acc's sources: "+",".join(s.getName() for s in accumulator.subscriptions.keys()))
            Transformer.__init__(self,
                                 [accumulator],
                                 {accumulator.getName(): 'd'},
                                 lambda d: logger.debug(self.model.train_on_batch(X=d, y=d)))
            logger.debug("self sources: "+",".join(s.getName() for s in self.subscriptions.keys()))
        else:
            pass # Load saved model

#class CholletAutoEncoder(Transformer):
#    """
#    An alternative AE based on fchollet's example code
#    """
#    def __init__(self, source, input_dim, train):
#        if train:
#            self.batch_size = 15
#            self.epochs = 3
#            self.layer_sizes = [input_dim] + [512, 256, 128, 64, 32, 16]
#            first_batch = []
#            for i in range(self.batch_size):
#                source.push(self)
#                first_batch.append(np.array(source.pull()))
#                print(i)
#                time.sleep(1)
#            # Layer-wise pre-training
#            trained_encoders = []
#            X_tmp = first_batch
#            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
#                logger.debug("Pretraining: Input {} -> Output {}".format(n_in,n_out))
#                ae = Sequential()
#                enc = containers.Sequential([core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid')])
#                dec = containers.Sequential([core.Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')])
#                ae.add(core.AutoEncoder(encoder=enc, decoder=dec,
#                                        output_reconstruction=False, tie_weights=True))
#                ae.compile(loss='mse', optimizer='rmsprop')
#                ae.fit(X_tmp, X_tmp, batch_size=self.batch_size, nb_epoch=self.epochs)
#                trained_encoders.append(ae.layers[0].encoder)
#                X_tmp = ae.predict(X_tmp)
#            # Further training
#            self.model = Sequential()
#            for encoder in trained_encoders:
#                self.model.add(encoder)
#            self.model.add(core.Dense(input_dim=self.layer_sizes[-1],
#                                      output_dim=self.layer_sizes[0],
#                                      activation='sigmoid'))
#            self.model.add(core.Dense(input_dim=self.layer_sizes[0],
#                                      output_dim=self.layer_sizes[0],
#                                      activation='linear'))
#            for qmins in xrange(4):
#                for i in xrange(15):
#                    source.push(self)
#
#        else:
#            pass

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

    def __init__(self
                ,input_dim
                ,bw_source
                ,ph_source = Source(lambda: None)
                ,mode = TRAINING
                ,batch_size = 16
                ,epochs = 30
                ,weights_name = 'latest'):
        if mode is AutoTransformer.TUNING: raise ValueError("Can't instantiate an AT in 'tuning' mode")
        self.mode = mode.instantiate(self, [bw_source, ph_source]) # Source order matters!
        self.batch_size = batch_size
        self.epochs = epochs
        self.bw_source = bw_source
        self.ph_source = ph_source
        self.layer_sizes = [input_dim] + [64, 32, 16, 8]
        self.enc_decs = []
        self.current_batch = [[] for i in range(batch_size)]
        self.past_batches = []
        self.current_phase = phase_names[0]
        self.batched = 0
        self.model = None
        self.weight_file = self.model_dir + self.prefix + weights_name
        self.maxes = self.get_from_catalog("maxes", weights_name) or np.ones(input_dim)
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

    def training_transform(self, bw, ph):
        self.tuning_transform(bw, ph)
        if self.batched == self.batch_size:
            #print(self.current_batch)
            X_l = np.array(map(np.array, self.current_batch))
            for (lay, ae) in enumerate(self.enc_decs):
                loss = ae.train_on_batch(X_l, X_l)
                logger.info({"training_loss_pre_"+str(lay): {self.getName(): loss}})
                X_l = ae.predict(X_l, batch_size=self.batch_size, verbose=0)
            self.past_batches.append((self.current_batch, self.current_phase))
            # Note: this means that batches where the phase changes at the end are tagged as the new phase,
            # not the old one... Which is probably wildly inaccurate.
            self.batched = 0
            return X_l
        else:
            return None
            # Is this necessary or even useful?
            #x_l = np.array(bw)
            #for ae in self.enc_decs:
            #    x_l = ae.predict(x_l, batch_size=1, verbose=0)
            #return x_l

    def tuning_transform(self, bw, ph):
        self.maxes = np.maximum(self.maxes, np.abs(bw))
        self.current_batch = self.current_batch[1:]+[np.divide(bw, self.maxes)]
        self.batched += 1
        self.current_phase = ph
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

    #TODO make this run in a separate thread? See to it that all incoming signals are handled correctly during
    def finetune(self):
        self.change_mode(self.TUNING)
        if not self.model:
            self.new_model()
        batches, phases = zip(*self.past_batches)
        X_train, X_test, y_train, y_test = train_test_split(
            map(np.array, batches),
            np_utils.to_categorical(map(
                lambda n: phase_names.index(n),
                phases)), #[p for p in phases for i in range(self.batch_size)])),
            test_size=0.1
        )
        print(type(np.array(y_train)))
        self.model.fit(np.array(X_train), y_train, batch_size=self.batch_size, nb_epoch=self.epochs, #GUESS
                       show_accuracy=bool(self.enc_decs), validation_data=(X_test, y_test))
        score = self.model.evaluate(X_test, y_test, show_accuracy=bool(self.enc_decs), verbose=0)
        logger.info({"finetune_score": score})
        self.change_mode(self.USING)

    def change_mode(self, new_mode):
        self.mode = new_mode.instantiate(self, [self.bw_source, self.ph_source])
        self.transform = self.mode.transform
        self.arg_assigns = self.mode.t_assignments
        self.no_updates = self.mode.no_updates
        # DIRTY
        self.setSources(self.mode.sources)

    def new_encdecs(self, compile = True):
        self.enc_decs = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            ae = Sequential()
            enc = containers.Sequential([core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid')])
            dec = containers.Sequential([core.Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')])
            ae.add(core.AutoEncoder(encoder=enc, decoder=dec,
                                    output_reconstruction=False))
            if compile:
                ae.compile(loss='mse', optimizer='rmsprop')
            self.enc_decs.append(ae)

    def new_model(self, fresh = False, compile = True):
        self.model = Sequential()
        if self.enc_decs and not fresh:
            for enc in [ae.layers[0].encoder for ae in self.enc_decs]:
                self.model.add(enc)
        else:
            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
                self.model.add(core.Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid'))
                #TODO ?
        self.model.add(core.Dense(input_dim=self.layer_sizes[-1]
                                  ,output_dim=len(phase_names)
                                  ,activation='softmax'))
        if compile:
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def load_model(self, f_name = None):
        if not self.model() and not f_name:
            self.new_model()
        elif f_name:
            if not self.model():
                f_name_sizes = self.get_from_catalog("layer_sizes", f_name)
                latest_sizes = self.get_from_catalog("layer_sizes")
                self.layer_sizes = f_name_sizes or latest_sizes
                self.new_model(fresh = True, compile = False)
            self.weight_file = self.model_dir + self.prefix + f_name
        else:
            pass #?
        self.model.load_weights(self.weight_file)

    def save_model(self, f_name = None):
        # TODO save in catalog
        if not self.model:
            logger.error({"error": {self.getName(): "Tried to save, but no model!"}})
        else:
            name = (self.model_dir + self.prefix + f_name) if f_name else self.weight_file
            self.model.save_weights(name, overwrite=True)

    def save_encdecs(self, f_name = None):
        assert len(self.enc_decs) > 0
        for (i, ed) in enumerate(self.enc_decs):
            base = (self.model_dir + self.prefix + f_name) if f_name else self.weight_file
            name = base + '_ed_' + str(i)
            ed.save_weights(name, overwrite=True)

    # TODO switch to using self.weight_file
    def load_encdecs(self, f_name = None):
        if not self.enc_decs and not f_name:
            self.new_encdecs(compile=False)
        elif f_name:
            self.layer_sizes = self.get_from_catalog("layer_sizes", f_name) or self.get_from_catalog("layer_sizes")
            self.new_encdecs(compile=False)
        for (i, ed) in enumerate(self.enc_decs):
            base = (self.model_dir + self.prefix + f_name) if f_name else self.weight_file
            name = base + '_ed_' + str(i)
            ed.load_weights(name)
            ed.compile(loss='categorical_crossentropy', optimizer='rmsprop')



# TODO write tests for AutoTransformer
# TODO train an AT on previous data
# TODO find out where the increasing training loss after restoring weights comes from
if __name__ == '__main__':
    l = LogSourceMaker()
    b = l.get_block()
    bws = b.sources[0]
    phs = b.sources[1]
    prep = Preprocessing(bws)
    logger.info("prep output dim: "+str(prep.output_dim))
    ae = AutoTransformer(prep.output_dim, prep, phs)
    print('have ae')
    #ae.load_encdecs()
    #sink = Sink([ae], lambda r: print(r) if (not r is None) else None)
    #print('have sink')
    b.start()
    time.sleep(180)
    b.stop()
    print('stopped')
    ae.save_encdecs()
    ae.finetune()
    print('finetuned')
    ae.save_model()
    print('saved')

