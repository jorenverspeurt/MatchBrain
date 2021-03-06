# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import math
import operator
import re
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as mpp
import seaborn as sns

import plotly.graph_objs as gob
import plotly.offline as ply
from plotly import tools as plt
from pymonad.List import *
from pymonad.Maybe import *

sns.set(style="whitegrid")
sns.set_palette('bright')

notebook_mode = False
notebook_mode_init = False

iden = lambda x: x
true = lambda _: True
labels = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']

def missing_attributes(dct, names):
    missing = []
    for name in names:
        if name not in dct.iterkeys():
            missing.append(name)
    if missing:
        return ", ".join(missing)+" were not found in "+str({str(k): Ell() for k in dct.iterkeys()})
    else:
        return ""

class MyList(List):
    def getValue(self):
        return list(self)

class Ell(object):
    def __str__(self):
        return "..."
    def __repr__(self):
        return "..."

class Nothingable(object):
    def __init__(self, value, nothing = False):
        self.value = value
        self.nothing = nothing

    def __nonzero__(self):
        return not self.nothing


class Averaging(Functor):
    # Averages its value along the 0 axis when retrieving it
    # This probably violates functor laws
    def fmap(self, function):
        # Pass along the fmap to whatever is inside
        return Averaging(function * self.value)
    def getValue(self):
        res = self.value.getValue()
        if res:
            return list(np.mean(res, axis=0))
        else:
            return None


class AdrGettable(Nothingable):
    default_attr = 'value'

    def __init__(self, name, value, copy_dct=None, nothing=False):
        super(AdrGettable, self).__init__(value, nothing)
        self.traversables = ['value']
        if copy_dct and not nothing:
            self.__dict__ = copy_dct
        self.name = name

    def get(self, adrs):
        @curry
        def getValue(o):
            return o.getValue()
        # handle_* :: str -> F obj
        @curry
        def handle_parens(cont, path):
            # for example bla/(hey/foo)(bing/baz&biz)(*/buzz)/bing
            # nested should work /bla/(hey/(foo/bar)(foe/bae))(...
            if '(' in path:
                def take_until_balanced(rem):
                    open = 0
                    closed = 0
                    front = ""
                    for c in rem:
                        if c == '(':
                            if open > 0:
                                front = front + c
                            open += 1
                        elif c == ')':
                            closed += 1
                            if open > closed:
                                front = front + c
                        else:
                            front = front + c
                        if open == closed:
                            break
                    return front, rem[len(front)+2:]

                front, remaining = path.split('(', 1)
                subs = []
                remaining = '('+remaining # whatevs
                while remaining.startswith('('):
                    sub, remaining = take_until_balanced(remaining)
                    subs.append(sub)
                fulls = map(lambda s: front+s+remaining, subs)
                return getValue * handle_parens(cont) * MyList(*fulls) # prevent stacked functors
            else:
                return cont(path)

        @curry
        def handle_slash(cont, path):
            # perform a recursive get
            if '/' in path:
                ht = []; ol = len(path)+1; nl = 0
                # Keep splitting one '/' until we have either just a head or a head and a tail (so ///foo//bar works)
                while len(ht) < 2 and (ol != nl):
                    ol = nl
                    ht = filter(None, path.split('/', 1))
                    nl = len(''.join(ht))
                if len(ht) == 2:
                    @curry
                    def rec_get(a, o):
                        if hasattr(o, 'get'):
                            return o.get(a)
                        elif hasattr(o, 'fmap'):
                            return rec_get(a) * o
                        # elif isinstance(o, list):
                        #     return o #HACK
                        else:
                            raise ValueError("Rec_get doesn't know what to do with this shit")
                    return rec_get(ht[1]) * cont(ht[0]) #fmap
                elif len(ht) == 1:
                    return cont(ht[0])
                else:
                    return cont('') # The string was just something like / or // ...
            else:
                return cont(path)

        @curry
        def handle_amp(cont, path):
            # multiple selection
            if '&' in path:
                selections = path.split('&')
                return getValue * cont * MyList(*selections) # prevent stacked functors
            else:
                return cont(path)

        @curry
        def handle_star(cont, path):
            # if the path to get is * then get the rest over all traversables
            # TODO make this work like shell globbing (i.e. /models/final-test-*)
            if path == "*":
                return getValue * cont * MyList(*list(sorted(self.traversables))) # prevent stacked functors
            else:
                return cont(path)

        @curry
        def handle_percent(cont, path):
            # like '*' but averages results, just for convenience, limited usefulness
            # TODO see handle_star
            if path == "%":
                return getValue * cont * Averaging(MyList(*self.traversables))
            else:
                return cont(path)

        @curry
        def handle_bang(cont, path):
            # index operator
            if '!' in path:
                dexes = path.split('!') # Only safe if this is really late in the chain!
                aname, indexes = dexes[0], map(int, dexes[1:])
                result = cont(path)
                @curry
                def safe_index(i, l):
                    if isinstance(l, list) and len(l) > 0:
                        return Just(l[i])
                    else:
                        return Nothing
                return reduce(lambda r, i: r >> safe_index(i), indexes, Just(result))
            else:
                return cont(path)

        @curry
        def handle_finally(path):
            if path == '':
                return Just(self)
            elif path in self.__dict__:
                return Just(self.__dict__[path])
            else:
                return Nothing

        result = handle_parens(handle_slash(handle_amp(handle_star(handle_percent(handle_finally)))), adrs)
        return result.getValue()

    # TODO add a "number of arguments" attribute and make this thing generic in that as well
    def bond(self, attr = None, transf = iden, cond = true):
        attr = attr or self.default_attr
        if not self.nothing:
            sdc = dict(self.__dict__)
            sdc[attr] = transf(sdc[attr])
            if cond(sdc[attr]):
                return type(self)(None, copy_dct = sdc, nothing = False)
            else:
                return type(self)(None, copy_dct = None, nothing = True)
        else:
            return self

    def mop(self, fun):
        return self.bond(fun.exp_attr if hasattr(fun, 'exp_attr') else None, transf = fun)

    def folter(self, fun):
        return self.bond(fun.exp_attr if hasattr(fun, 'exp_attr') else None, cond = fun)

    def _repr(self, parentlist = ()):
        return {kv[0]: kv[1]._repr(parentlist+(self,))
                       if hasattr(kv[1], '_repr')
                       else kv[1]
                for kv in filter(lambda t: '_' not in t[0],
                                 self.__dict__.iteritems())
                if kv[1] not in parentlist}

    def __repr__(self):
        # Dirty, so dirty
        return str(self._repr(()))


class AGD(AdrGettable):
    def __init__(self, name, value):
        super(AGD, self).__init__(name, None)
        self.traversables = value.keys()
        self.__dict__.update(value)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return len(self.traversables)

def list_to_agd(name, l):
    return AGD(name, {i: l for i,l in enumerate(l)})


class Catalog(AdrGettable):
    def __init__(self, name, dct, copy_dct=None, nothing=False):
        super(Catalog, self).__init__(name, dct, copy_dct, nothing)
        if nothing:
            self.experiments = None
        elif not copy_dct:
            main_keys = {k for k in dct.iterkeys() if not (k.startswith('e-') or k.startswith('m-'))}
            self.traversables = list(main_keys)
            self.__dict__.update({k: Experiment(k,
                                                dct[k],
                                                AGD(name+'_models',
                                                    {mk: Model(mk, dct[mk])
                                                     for mk in dct.iterkeys()
                                                     if mk.startswith('m-' + k)}),
                                                AGD(name+'_encdecs',
                                                    {ek: EncDecs(ek, dct[ek])
                                                     for ek in dct.iterkeys()
                                                     if ek.startswith('e-' + k)}))
                                  for k in main_keys})

class Experiment(AdrGettable):
    default_attr = 'summary'

    def __init__(self, name, summary, model_dict, ed_dict, copy_dct=None, nothing=False):
        super(Experiment, self).__init__(name, summary, copy_dct, nothing)
        if nothing:
            self.traversables = []
            self.cross_validation = None
            self.settings = None
            self.results = None
            self.enc_decs = None
            self.models = None
        elif not copy_dct:
            self.traversables = ['cross_validation', 'settings', 'results', 'encdecs', 'models']
            ma = missing_attributes(summary, ['cross_validation', 'settings'])
            if ma:
                raise ValueError("In "+self.name+", "+ma)
            self.cross_validation = summary['cross_validation']
            self.settings = ExperimentSettings(name+"_settings", summary['settings'])
            self.results = ExperimentResults(name+"_results", {k: v for k, v in summary.iteritems() if k.startswith('-')}, self)
            self.enc_decs = ed_dict
            self.models = model_dict

    def bond(self, attr = 'summary', transf = iden, cond = true):
        if not self.nothing:
            sdc = dict(self.__dict__)
            sdc[attr] = transf(sdc[attr])
            if cond(sdc[attr]):
                return Experiment(self.name, None, None, None)
            else:
                return Experiment(self.name, None, None, None)
        else:
            return self


class ExperimentSettings(AdrGettable):
    def __init__(self, name, dct, copy_dct=None, nothing=False):
        super(ExperimentSettings, self).__init__(name, dct, copy_dct, nothing)
        if not nothing:
            find_attrs = map(lambda r: any(map(lambda k: re.match('\d-'+r, k)
                                              ,dct.iterkeys()))
                            ,['dr', '\(gbs,gsf\)', 'cl', 'co', 'eo', 'l1s', 'l2s'])
            ma = not all(find_attrs)
            if ma:
                print(find_attrs)
                raise ValueError("Settings missing attributes: "+str({str(k): Ell() for k in dct.iterkeys()}))
            self.traversables = ['drop_rates', 'gauss_combs', 'class_losses', 'class_opts', 'encdec_opts', 'l1s', 'l2s']
            stripped = {k.split('-',1)[1]: v for k,v in dct.iteritems()}
            self.drop_rates = stripped['dr']
            self.gauss_combs = stripped['(gbs,gsf)']
            self.class_losses = stripped['cl']
            self.class_opts = stripped['co']
            self.encdec_opts = stripped['eo']
            self.l1s = stripped['l1s']
            self.l2s = stripped['l2s']


class ExperimentResults(AdrGettable):
    def __init__(self, name, dct, parent = None, copy_dct=None, nothing=False):
        super(ExperimentResults, self).__init__(name, dct, copy_dct, nothing)
        self.parent = parent
        if dct and not nothing and not copy_dct:
            self.traversables = ['_t_summaries']
            # The experiment's results for all completed cross-validation folds are accessible via this object
            # (So non-completed folds are effectively hidden)
            # If no cross-validation was performed (or only 1 fold was ever started) then all results are included
            # TODO check per option so that a consistent subset of executions is visible (largest complete subtree?)
            # i.e. 0000, 0001, 0002, 0010, 0011, 0012, 0100, 0101 => 0000, 0001, 0002, 0010, 0011, 0012
            # should be the largest number 0, 1, ..., len(options[-1])*1, len(options[-1])*2, len(options[-1])*len(options[-2]), ...
            # that is still smaller than len(executions)
            attempted_cross_is = dct and 'x' in dct.keys()[0] and {int(k[k.find('x')+1]) for k in dct.iterkeys()}
            last_completed = -1
            if not parent: #Shouldn't really happen...
                was_cross = attempted_cross_is and len(attempted_cross_is) > 1
                if was_cross:
                    runs_per_xval = max(len(value) for value in dct.itervalues())
                    last_completed = max([i
                                          for i in attempted_cross_is
                                          if len([v
                                                  for k,v in dct.iteritems()
                                                  if 'x'+str(i) in k][0])
                                             == runs_per_xval]+[-1])
                    self.cross_validation_started = True
                    # May be wrong
                    self.cross_validation_finished = last_completed == max(attempted_cross_is)
                else:
                    self.cross_validation_started = False
                    self.cross_validation_finished = None
            else:
                was_cross = parent.cross_validation['enabled']
                self.cross_validation_started = bool(attempted_cross_is) and was_cross
                self.cross_validation_finished = bool(attempted_cross_is) and (len(attempted_cross_is) ==
                                                                               reduce(operator.mul
                                                                                     ,map(len
                                                                                         ,parent.settings.value.itervalues())
                                                                                     ,1))
                last_completed = (attempted_cross_is and len(attempted_cross_is)>1 and ((self.cross_validation_finished and max(attempted_cross_is)) or max(attempted_cross_is-{max(attempted_cross_is)}))) or (attempted_cross_is and attempted_cross_is.pop()) or None
            if self.cross_validation_started:
                isolate_setting = lambda s: filter(lambda subs: re.match('^\d', subs), s.split('-'))[0]
                self.summaries = [{isolate_setting(fk): fv
                                   for fk,fv in filter(lambda t: 'x'+str(i) in t[0], dct.iteritems())}
                                  for i in attempted_cross_is
                                  if i <= last_completed]
            else:
                self.summaries = [{k[1:]: v for k,v in dct.iteritems()}] # Cut off the '-' at the start
            accuracies = [r['test_accuracy'] if 'test_accuracy' in r else None for xv in self.summaries for r in xv.itervalues() if 'finished' in r and r['finished'] and r['test_accuracy']]
            self.average = np.mean(accuracies) if len(accuracies) else None
            self.stddev = np.std(accuracies) if len(accuracies) else None
        if not dct:
            raise ValueError("Tried to construct ERs for "+("?" if not parent else parent.models[parent.model.keys()[0]][2:])+" but no results were passed!")

    @property
    def _t_summaries(self):
        return list_to_agd(self.name+'_summaries', self.summaries)

    def bond(self, attr = 'accuracy', transf = iden, cond = true):
        if not self.nothing:
            sdc = dict(self.__dict__)
            sdc[attr] = transf(sdc[attr])
            if cond(sdc[attr]):
                return ExperimentResults(self.name, None, self.parent, sdc)
            else:
                return ExperimentResults(self.name, None, self.parent, None)
        else:
            return self

    def by_xval(self):
        return {i: np.mean([v['test_accuracy']
                            for v in self.summaries[i].itervalues()])
                for i in xrange(len(self.summaries))}

    def by_settings(self):
        return {k: np.mean([self.summaries[i][k]['test_accuracy']
                            for i in xrange(len(self.summaries))])
                for k in self.summaries[0].iterkeys()}
    #TODO: do something with times (total durations, ... ?)


class Model(AdrGettable):
    default_attr = 'accuracy'
    dataframe_columns = ['model_name', 'label', 'training_set_size' # per-model values
                        ,'epochs', 'class_optimizer', 'class_loss'
                        ,'noise_base_sigma', 'noise_sigma_factor' ,'drop_rate'
                        ,'pretraining_duration'
                        ,'l1_normalization', 'l2_normalization'
                        ,'test_accuracy', 'test_loss' ] + \
                        [l+'_'+s # per-label values
                         for s in ['loss', 'accuracy'
                                  ,'recall', 'miss_rate', 'fallout', 'specificity'
                                  ,'precision', 'false_omission', 'false_discovery', 'npv']
                         for l in labels] + \
                        [l_from+'_predicted_'+l_to # cross-label values
                         for l_from in labels
                         for l_to in labels]

    def __init__(self, name, dct, copy_dct=None, nothing=False):
        super(Model, self).__init__(name, dct, copy_dct, nothing)
        if dct and not nothing and not copy_dct:
            ma = missing_attributes(dct, ['class_optimizer', 'class_loss', 'gaussian_base_sigma', 'gaussian_sigma_factor', 'drop_rate', 'pretrain_start_time', 'pretrain_end_time', 'labels', 'epochs', 'test_loss', 'test_accuracy'])
            if ma:
                raise ValueError(ma)
            self.traversables = ['classification', 'noise', 'pretraining', 'losses', 'counts', 'confusions']
            self.xval_number = int(name[name.find('-x')+2])
            def try_int(char):
                try:
                    int(char)
                    return True
                except Exception:
                    return False

            self.settings_number = filter(lambda ss: all(try_int(c) for c in ss), name.split("-"))[0]
            self.classification = AGD(self.name+'_class', {'optimizer': dct['class_optimizer'], 'loss': dct['class_loss']})
            self.noise = {'base_sigma': dct['gaussian_base_sigma'], 'sigma_factor': dct['gaussian_sigma_factor']}
            self.drop_rate = dct['drop_rate']
            time_str_format = "%y%m%d-%H%M%S"
            self.pretraining = AGD(self.name+'_pretraining',
                                   {'start': datetime.strptime(dct['pretrain_start_time'], time_str_format)
                                   ,'end':   datetime.strptime(dct['pretrain_end_time'], time_str_format)})
            self.pretraining_duration = (self.pretraining['end']-self.pretraining['start']).seconds
            self.epochs = dct['epochs']
            self.normalization = AGD(self.name+'_normalization', {'l1': dct['l1'], 'l2': dct['l2']})
            self.accuracy = dct['test_accuracy']
            self.loss = dct['test_loss']
            self.losses = AGD(self.name+'_losses', {k: v['loss'] for k,v in dct['labels'].iteritems()})
            self.counts = AGD(self.name+'_counts', {k: v['counts'] for k,v in dct['labels'].iteritems()})
            self.train_size = sum(l for v in self.counts.get('*') for l in v)
            c = lambda x,y: self.counts[x][y]
            li = lambda k: labels.index(k)
            lr = lambda: xrange(len(labels))
            # confusion counts
            cc = {k: [    c(k,li(k))
                     ,sum(c(k,i) for i in lr() if i != li(k))
                     ,sum(c(j,li(k)) for j in labels if j != k)
                     ,sum(c(j,i) for i in lr() if i != li(k) for j in labels if j != k)]
                  for k in labels}
            # TODO other metrics
            pzd = lambda n: 1 if n == 0 else n
            self.confusions = AGD(self.name+'_confusions',
                                  {'counts': cc
                                  ,'condition_rates':  AGD(
                                      self.name+'_condition_rates',
                                      {k: [float(cc[k][0])/pzd(cc[k][0]+cc[k][1])
                                          ,float(cc[k][1])/pzd(cc[k][1]+cc[k][0])
                                          ,float(cc[k][2])/pzd(cc[k][2]+cc[k][3])
                                          ,float(cc[k][3])/pzd(cc[k][3]+cc[k][2])]
                                       for k in labels})
                                  ,'prediction_rates': AGD(
                                      self.name+'_prediction_rates',
                                      {k: [float(cc[k][0])/pzd(cc[k][0]+cc[k][2])
                                          ,float(cc[k][1])/pzd(cc[k][1]+cc[k][3])
                                          ,float(cc[k][2])/pzd(cc[k][2]+cc[k][0])
                                          ,float(cc[k][3])/pzd(cc[k][3]+cc[k][1])]
                                       for k in labels})
                                  })

    def to_row(self):
        row = {
            'model_name': self.name, 'training_set_size': self.train_size, 'epochs': self.epochs,
            'class_optimizer': self.classification['optimizer'], 'class_loss': self.classification['loss'],
            'noise_base_sigma': self.noise['base_sigma'], 'noise_sigma_factor': self.noise['sigma_factor'],
            'drop_rate': self.drop_rate, 'pretraining_duration': self.pretraining_duration,
            'l1_normalization': self.normalization['l1'], 'l2_normalization': self.normalization['l2'],
            'test_accuracy': self.accuracy, 'test_loss': self.loss
        }
        row.update({l+'_'+s[0]: s[1]
                    for l in labels
                    for s in [('loss', self.losses[l])
                             ,('accuracy', (self.confusions['counts'][l][0]+self.confusions['counts'][l][3])/self.train_size)
                             ,('recall',          self.confusions['condition_rates'][l][0])
                             ,('miss_rate',       self.confusions['condition_rates'][l][1])
                             ,('fallout',         self.confusions['condition_rates'][l][2])
                             ,('specificity',     self.confusions['condition_rates'][l][3])
                             ,('precision',       self.confusions['prediction_rates'][l][0])
                             ,('false_omission',  self.confusions['prediction_rates'][l][1])
                             ,('false_discovery', self.confusions['prediction_rates'][l][2])
                             ,('npv',             self.confusions['prediction_rates'][l][3])]
        })
        row.update({l_from+'_predicted_'+l_to: self.counts[l_from][i]
                    for l_from in labels
                    for (i, l_to) in enumerate(labels)
        })
        if not len(row) == len(self.dataframe_columns):
            raise ValueError(len(row), len(self.dataframe_columns))
        return row

#TODO write dataframe conversions for experiments and possibly encdecs


class EncDecs(AdrGettable):
    def __init__(self, name, dct, copy_dct=None, nothing=False):
        super(EncDecs, self).__init__(name, dct, copy_dct, nothing)
        if dct and not copy_dct and not nothing:
            ma = missing_attributes(dct, ['layers', 'encdec_optimizer', 'layer_sizes', 'quality', 'train_quality', 'gaussian_base_sigma', 'gaussian_sigma_factor', 'drop_rate', 'l1', 'l2', 'epochs'])
            if ma:
                raise ValueError(ma)
            self.traversables = ['_t_layers', '_t_sizes', 'noise', 'normalization']
            self.layers = [v for (_,v) in sorted(dct['layers'].iteritems())]
            self.optimizer = dct['encdec_optimizer']
            self.sizes = dct['layer_sizes']
            self.quality = dct['quality']
            self.train_quality = dct['train_quality']
            self.noise = AGD(self.name+'_noise', {'base_sigma': dct['gaussian_base_sigma']
                                                 ,'sigma_factor': dct['gaussian_sigma_factor']})
            self.drop_rate = dct['drop_rate']
            self.normalization = AGD(self.name+'_normalization', {'l1': dct['l1'], 'l2': dct['l2']})
            self.epochs = dct['epochs']

    @property
    def _t_layers(self):
        return list_to_agd(self.name+'_layers', self.layers)

    @property
    def _t_sizes(self):
        return list_to_agd(self.name+'_sizes', self.sizes)


def exp_attr(name):
    def wrapper(func):
        setattr(func, 'exp_attr', name)
        return func
    return wrapper

#TODO bring in training history

class Tester(object):
    class PrintableOf(object):
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return "%s(%s)" % (type(self).__name__, self.value)

    class Success(PrintableOf):
        def __init__(self, value = ()):
            super(Tester.Success, self).__init__(value)
            self.kind = 'success'

        def bind(self, fun):
            return fun(self.value)

        def __repr__(self):
            return u"✓"

        def __nonzero__(self):
            return True

    class Errored(PrintableOf):
        def __init__(self, value, message = ""):
            super(Tester.Errored, self).__init__(value)
            self.message = message
            self.fail_messages = []
            self.kind = 'errored'

        def bind(self, _):
            return self

        def __repr__(self):
            if not self.fail_messages:
                return super(Tester.Errored, self).__repr__()
            else:
                name = self.value.name if hasattr(self.value, 'name') else str(self.value)
                return "Errored(%s, %s, [%s])" % (name, self.message, ("\n"+(" "*(8+len(name)+2+len(self.message)+2))+",").join(self.fail_messages)+"]")

    class Failed(PrintableOf):
        def __init__(self, value, message = ""):
            super(Tester.Failed, self).__init__(value)
            self.fail_messages = [message]
            self.kind = 'failed'

        def bind(self, fun):
            r = fun(self.value)
            if r.kind == 'success':
                result = Tester.Failed(r.value)
                result.fail_messages = self.fail_messages
            elif r.kind == 'errored':
                result = Tester.Errored(r.value)
                result.fail_messages = self.fail_messages
            elif r.kind == 'failed':
                result = Tester.Failed(r.value, r.fail_messages[0])
                result.fail_messages = self.fail_messages + result.fail_messages
                return result

        def __repr__(self):
            return "Failed(%s)" % (("\n"+(" "*7)+";").join(self.fail_messages))


    def __init__(self, value):
        self.initial = Tester.Success(value)
        self.todo = {}
        self.done = {}
        self.order = []

    def do(self, name, fun, uses = None):
        if uses == None or (uses in self.todo.keys() or self.done.keys()):
            self.todo[name] = (uses, fun)
            self.order.append(name)
            return self
        else:
            raise ValueError("That dependency is not available!")

    def run(self):
        for name in self.order:
            (dep, fun) = self.todo[name]
            value = self.initial if dep == None else self.done[dep]
            self.done[name] = value.bind(fun)
        self.order = []
        return map(lambda t: t[1], sorted(self.done.iteritems()))


def flatten(l, num=1):
    if num == 0:
        return l

    new_l = []
    for item in l:
        if type(item) == type([]):
            new_l.extend(flatten(item, num-1))
        else:
            new_l.append(item)
    return new_l

class Plotter(object):
    def __init__(self, catalog):
        self.catalog = catalog

    def _handle_args(self, groups, x_data, x_path, y_data, y_paths, flatten_y):
        assert x_data or x_path and not (x_data and x_path)
        assert y_data or y_paths and not (y_data and y_paths)
        if x_data:
            x = x_data
        else:
            x = self.catalog.get(x_path)
        if y_data:
            y = y_data
        else:
            y = flatten(map(self.catalog.get, y_paths), flatten_y)
            while len(y) == 1 and isinstance(y[0], list):
                y = y[0]
        if len(y) != len(groups):
            if len(y) == len(x)*len(groups):
                y = [y[i: i+len(x)] for i in xrange(len(groups))]
            else:
                raise ValueError("Length of y doesn't match up! Expected "+str(len(groups))+" or "+str(len(x)*len(groups))+" but got "+str(len(y)))
        return x, y


class PlyPlotter(Plotter):
    def __init__(self, catalog):
        super(PlyPlotter, self).__init__(catalog)
        self.multi = False
        self.multi_figures = []
        self.multi_titles = []
        self.multi_columns = 1
        self.multi_name = ""

    def start_multi(self, name = "", cols = 1):
        self.multi = True
        self.multi_name = name
        self.multi_columns = cols

    def _plot(self, grobby, filename):
        global notebook_mode, notebook_mode_init
        if notebook_mode:
            if not notebook_mode_init:
                ply.init_notebook_mode()
                notebook_mode_init = True
            ply.iplot(grobby)
        else:
            ply.plot(grobby, filename=filename)

    def _append(self, traces, name):
        self.multi_figures.append(traces)
        self.multi_titles.append(name)

    def bar(self, fig_name, groups = (), x_data = None, x_path = None, y_data = None, y_paths = None, flatten_y = 0):
        x, ys = self._handle_args(groups, x_data, x_path, y_data, y_paths, flatten_y)
        traces = map(lambda (y, name): gob.Bar(x = x, y = y, name = name, orientation = 'v'),
                     [(ys[i], groups[i]) for i in xrange(len(groups))])
        if len(groups)>1:
            layout = gob.Layout(barmode = 'group', title = fig_name)
        else:
            layout = gob.Layout(title = fig_name)
        fig = gob.Figure(data=traces, layout=layout)
        if not self.multi:
            self._plot(fig, fig_name)
        else:
            self._append(fig, fig_name)

    def multi_plot(self, width = 2480, height = 3508):
        # Print A4 in px: 2480x3508
        rows = int(math.ceil(float(len(self.multi_figures))/self.multi_columns))
        # Reminder: it's possible to make some figures span multiple columns, probably not going to use that though
        figure = plt.make_subplots(rows=rows, cols=self.multi_columns, subplot_titles=self.multi_titles)
        for i in xrange(len(self.multi_figures)):
            figure.append_trace(self.multi_figures[i], (i%self.multi_columns)+1, (i%rows)+1)
        figure['layout'].update(width=width, height=height, title=self.multi_name)
        self._plot(figure, self.multi_name)

class SeabornPlotter(Plotter):
    def bar(self, fig_name, groups = (), x_data = None, x_path = None, y_data = None, y_paths = None, flatten_y = 0):
        x, ys = self._handle_args(groups, x_data, x_path, y_data, y_paths, flatten_y)
        df = pd.DataFrame()
        for xi in x:
            df[xi] = None

    def model_bar(self, runs,
                  x, y_type, y,
                  cross_label_from = 'row', cross_label_to = 'x',
                  hue = 'experiment', # must be either experiment or a per-model value, ignored if cross_label_ANY == 'hue'
                  row_type = 'per-model', row = None, # ignored if x_type == 'cross_label' and cross_label_ANY == 'row'
                  col_type = 'per_model', col = None, # ignored if x_type == 'cross_label' and cross_label_ANY == 'col'
                  run_aliases = None # alternative experiment names
        ):
        class EqToAll(object):
            def __eq__(self, other):
                return True
        opts = lambda l: sum(("("+li+")" for li in l), "")
        models = self.catalog.get(opts(runs) + "/models/*")
        rows = map(lambda r: map(lambda m: m.to_row(), r), models)
        run_names = run_aliases or runs
        for (i,r) in enumerate(run_names):
            for model_row in rows[i]:
                model_row['experiment'] = r
        if y_type == 'cross-label':
            # TODO don't assume that x is a label axis, let row-col be label axes for example
            if not 'label' in x:
                # x must be either actual_label or predicted_label
                raise ValueError("Wrong value for x!")
            parts = y.split('_')
            filt,_y = (parts[0],'_'.join(parts[1:])) \
                      if parts[0] in labels \
                      else ((parts[-1],'_'.join(parts[:-1]))
                            if parts[-1] in labels
                            else EqToAll())
            known_needed_keys = []
            if row and 'label' not in row: known_needed_keys.append(row)
            if col and 'label' not in col: known_needed_keys.append(col)
            if hue and 'label' not in hue: known_needed_keys.append(hue)
            new_rows = []
            for run_rows in rows:
                for model_row in run_rows:
                    new_rows += [dict({k: model_row[k]
                                       for k in known_needed_keys},
                                      **{'actual_label': l_from
                                        ,'predicted_label': l_to
                                        ,_y: model_row[l_from+'_'+l_to]})
                                 for l_from in labels
                                 for l_to in labels
                                 if (l_from == filt if x == 'predicted_label' else l_to == filt)]
            df = pd.DataFrame(flatten(rows))
            bars = sns.factorplot(data = df, x = x, y = _y, hue = hue, row = row, col = col, kind = 'bar', legend = False)
        elif y_type == 'per-label':
            parts = y.split('_')
            if any(map(lambda s: s == 'label', [x, row, col, hue])):
                # There's a label axis
                if any(map(lambda s: s in labels, parts)):
                    raise ValueError("Can't filter when there's already a label axis!")

            else:
                if any(map(lambda s: s in labels, parts)):
                    # Filtering needs to happen
                    filt =


        elif y_type == 'per-model':
            pass
        else:
            raise ValueError("Wrong value for x_type!")

        def labelled_transform(rtype, frame):
            result = {k: {} for k in Model.dataframe_columns if not rtype in k}
            result[rtype+'_label'] = {}
            result[rtype] = {}
            # itertuples is ~50x faster than iterrows
            for tup in frame.itertuples():
                for l in labels:
                    for (i, c) in enumerate(Model.dataframe_columns):
                        if rtype not in c:
                            result[c][tup[0]+'_'+l] = tup[i+1]
                        elif l in c:
                            result[rtype+'_label'][tup[0]+'_'+l] = l
                            result[rtype][tup[0]+'_'+l] = tup[i+1]
                        else:
                            pass # Turn every label+'_'+rtype column into 1 extra row
            return pd.DataFrame(result)

        if '_label' in x:
            df = labelled_transform('_'.join(x.split('_')[:-1]), df)
        df = pd.DataFrame(flatten(rows))
        bars = sns.factorplot(data = df, x = x, y = y, hue = hue, row = row, col = col, kind = 'bar', legend = False)
        bars.set_xticklabels(rotation=30)
        mpp.legend(loc='upper left')
        sns.plt.show()






if __name__ == '__main__':
    # Test
    def tw(fun, name = "", eq = None, desired = None, message = None):
        def wrapped(value):
            try:
                got = fun(value)
                if eq == None:
                    return Tester.Success(got)
                elif eq:
                    return Tester.Success(value) if got == desired else Tester.Failed(value, name + ": " + (message or "Expected %s, got %s") % (desired, got))
                else:
                    # Well, why not right?
                    return Tester.Success(value) if got != desired else Tester.Failed(value, name + ": " + (message or "Didn't expect %s") % (got,))
            except Exception as e:
                return Tester.Errored(e.message)
        return wrapped

    with open('catalog-merged-fixed.json', 'r') as f:
        catj = json.load(f)
    et = ['final-test-batch50']
    cato = Catalog("test", catj)
    tester = Tester(cato)
    print("TESTS\n=====")
    results = tester.\
        do(0, tw(lambda c: c.get(et[0]+'/results'))).\
        do(1, tw(lambda er: er.get('cross_validation_started'), 'xvs', True, True), 0).\
        do(2, tw(lambda er: er.cross_validation_started, 'xvs2', True, True), 0).\
        do(3, tw(lambda c: c.get(et[0]+'/settings'))).\
        do(4, tw(lambda es: es.get('drop_rates'), "drop_rates", True, [0.1]), 3).\
        do(5, tw(lambda es: es.drop_rates), 3).\
        do(6, tw(lambda es: len(es.gauss_combs), "gauss_combs", True, 1), 3).\
        do(7, tw(lambda es: es.name, "esname", True, et[0] + '_settings'), 3).\
        run()
    print(" "+" ".join(reduce(lambda old, new: old if len(old) and old[-1] == '\n' and new == '\n' else old+[new],
                              map(unicode,
                                  [a
                                   for r in results
                                   for a in ([r] if len(unicode(r))==1 else ['\n',r,'\n'])]),
                              [])))
    print("\nRESULTS\n=======")
    print(cato.get(et[0]+'/models/*/confusions/counts/INTENSE'))
    print(cato.get(et[0]+'/models/*/confusions/condition_rates&prediction_rates/INTENSE'))
    print(cato.get(et[0]+'/models/*/confusions/(condition_rates)(prediction_rates/INTENSE)'))
    print(cato.get('final-test-batch50()(-nodrop)(-nogauss)/models/%/counts/INTENSE'))
    print(cato.get('final-test-batch50/models/*/train_size'))
    # e_prefix = '(final-test-on)(final-test-off)(final-test-long)(final-test-batch50)(final-test-batch50-nodrop)(final-test-batch50-nogauss)/models/*'
    # intense_comp = map(lambda e: map(list, list(e)), list(cato.get(e_prefix+'/(noise/base_sigma)(drop_rate)(counts/DISTRACT&RELAXOPEN&RELAXCLOSED&CASUAL&INTENSE)(train_size)')))
    # indices = flatten(map(list, list(cato.get(e_prefix+'/name'))),3)
    # exp_aliases = ['on', 'off', 'long', 'batch50', 'batch50-nodrop', 'batch50-nogauss']
    # print(type(intense_comp))
    # pprint(intense_comp, width=200)
    # print(indices)



    print("\nPLOT\n====")
    if False:
        plotter = PlyPlotter(cato)
        for l in labels:
            plotter.bar("Intense Comparison",
                        groups=["batch50", "batch50-nodrop", "batch50-nogauss"],
                        x_data=labels,
                        y_paths=["final-test-batch50()(-nodrop)(-nogauss)/models/%/counts/"+l])
    else:
        plotter = SeabornPlotter(cato)
        plotter.model_bar([ 'final-test-on'
                          , 'final-test-off'
                          , 'final-test-long'
                          , 'final-test-batch50'
                          , 'final-test-batch50-nodrop'
                          , 'final-test-batch50-nogauss' ],
                          'predicted_label',
                          'accuracy',
                          row='label')


