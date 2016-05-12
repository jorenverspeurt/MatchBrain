from __future__ import print_function

import json
import operator
import re
from datetime import datetime

import numpy as np

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


class AdrGettable(Nothingable):
    default_attr = 'value'

    def __init__(self, value, copy_dct = None, nothing = False):
        super(AdrGettable, self).__init__(value, nothing)
        if copy_dct and not nothing:
            self.__dict__ = copy_dct

    def get(self, adrs):
        if '/' in adrs:
            head, tail = adrs.split('/', 1)
            return self.__dict__[head].get(tail)
        else:
            return self.__dict__[adrs]

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


class Catalog(AdrGettable):
    default_attr = 'experiments'

    def __init__(self, dct, copy_dct = None, nothing = False):
        super(Catalog, self).__init__(dct, copy_dct, nothing)
        if nothing:
            self.experiments = None
        elif not copy_dct:
            main_keys = {k for k in dct.iterkeys() if not (k.startswith('e-') or k.startswith('m-'))}
            self.experiments = {k: Experiment(dct[k],
                                              {mk: Model(dct[mk])
                                               for mk in dct.iterkeys()
                                               if mk.startswith('m-'+k)}
                                             ,{ek: EncDecs(dct[ek])
                                               for ek in dct.iterkeys()
                                               if ek.startswith('e-'+k)})
                                for k in main_keys}

    # WORNING: this one behaves differently from the rest...
    def bond(self, _ = 'experiments', transf = iden, cond = true):
        if not self.nothing:
            result = {k: transf(v) for k,v in self.experiments.iteritems() if cond(v)}
            if result:
                return Catalog(result, None, False)
            else:
                return Catalog(None, None, True)
        else:
            return self

class Experiment(AdrGettable):
    default_attr = 'summary'

    def __init__(self, summary, model_dict, ed_dict, copy_dct = None, nothing = False):
        super(Experiment, self).__init__(summary, copy_dct, nothing)
        if nothing:
            self.cross_validation = None
            self.settings = None
            self.results = None
            self.enc_decs = None
            self.models = None
        elif not copy_dct:
            ma = missing_attributes(summary, ['cross_validation', 'settings'])
            if ma:
                raise ValueError(ma)
            self.cross_validation = summary['cross_validation']
            self.settings = ExperimentSettings(summary['settings'])
            self.results = ExperimentResults({k:v for k,v in summary.iteritems() if k.startswith('-')}, parent = self)
            self.enc_decs = ed_dict
            self.models = model_dict

    def bond(self, attr = 'summary', transf = iden, cond = true):
        if not self.nothing:
            sdc = dict(self.__dict__)
            sdc[attr] = transf(sdc[attr])
            if cond(sdc[attr]):
                return Experiment(None, None, None, sdc)
            else:
                return Experiment(None, None, None, None, True)
        else:
            return self


class ExperimentSettings(AdrGettable):
    def __init__(self, dct, copy_dct = None, nothing = False):
        super(ExperimentSettings, self).__init__(dct, copy_dct, nothing)
        if not nothing:
            find_attrs = map(lambda r: any(map(lambda k: re.match('\d-'+r, k)
                                              ,dct.iterkeys()))
                            ,['dr', '\(gbs,gsf\)', 'cl', 'co', 'eo', 'l1s', 'l2s'])
            ma = not all(find_attrs)
            if ma:
                print(find_attrs)
                raise ValueError("Settings missing attributes: "+str({str(k): Ell() for k in dct.iterkeys()}))
            stripped = {k[2:]: v for k,v in dct.iteritems()} # Doesn't scale
            self.drop_rate = stripped['dr']
            self.gauss_combs = stripped['(gbs,gsf)']
            self.class_loss = stripped['cl']
            self.class_opt = stripped['co']
            self.encdec_opt = stripped['eo']
            self.l1s = stripped['l1s']
            self.l2s = stripped['l2s']


class ExperimentResults(AdrGettable):
    def __init__(self, dct, parent = None, copy_dct = None, nothing = False):
        super(ExperimentResults, self).__init__(dct, copy_dct, nothing)
        self.parent = parent
        if dct and not nothing and not copy_dct:
            # The experiment's results for all completed cross-validation folds are accessible via this object
            # (So non-completed folds are effectively hidden)
            # If no cross-validation was performed (or only 1 fold was ever started) then all results are included
            # TODO check per option so that a consistent subset of executions is visible (largest complete subtree?)
            # i.e. 0000, 0001, 0002, 0010, 0011, 0012, 0100, 0101 => 0000, 0001, 0002, 0010, 0011, 0012
            # should be the largest number 0, 1, ..., len(options[-1])*1, len(options[-1])*2, len(options[-1])*len(options[-2]), ...
            # that is still smaller than len(executions)
            attempted_cross_is = dct and 'x' in dct.keys()[0] and {int(k[k.find('x')+1]) for k in dct.iterkeys()}
            print(attempted_cross_is)
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
                print(last_completed)
                self.results = [{isolate_setting(fk): fv
                                 for fk,fv in filter(lambda t: 'x'+str(i) in t[0], dct.iteritems())}
                                for i in attempted_cross_is
                                if i <= last_completed]
            else:
                self.results = [{k[1:]: v for k,v in dct.iteritems()}] # Cut off the '-' at the start
            print(self.results)
            accuracies = [r['test_accuracy'] if 'test_accuracy' in r else None for xv in self.results for r in xv.itervalues() if 'finished' in r and r['finished'] and r['test_accuracy']]
            self.average = np.mean(accuracies) if len(accuracies) else None
            self.stddev = np.std(accuracies) if len(accuracies) else None
        if not dct:
            raise ValueError("Tried to construct ERs for "+("?" if not parent else parent.models[parent.model.keys()[0]][2:])+" but no results were passed!")

    def bond(self, attr = 'accuracy', transf = iden, cond = true):
        if not self.nothing:
            sdc = dict(self.__dict__)
            sdc[attr] = transf(sdc[attr])
            if cond(sdc[attr]):
                return ExperimentResults(None, self.parent, sdc, False)
            else:
                return ExperimentResults(None, self.parent, None, True)
        else:
            return self

    def by_xval(self):
        return {i: np.mean([v['test_accuracy']
                            for v in self.results[i].itervalues()])
                for i in xrange(len(self.results))}

    def by_settings(self):
        return {k: np.mean([self.results[i][k]['test_accuracy']
                            for i in xrange(len(self.results))])
                for k in self.results[0].iterkeys()}
    #TODO: do something with times (total durations, ... ?)


class Model(AdrGettable):
    default_attr = 'accuracy'

    def __init__(self, dct, copy_dct = None, nothing = False):
        super(Model, self).__init__(dct, copy_dct, nothing)
        if dct and not nothing and not copy_dct:
            ma = missing_attributes(dct, ['class_optimizer', 'class_loss', 'gaussian_base_sigma', 'gaussian_sigma_factor', 'drop_rate', 'pretrain_start_time', 'pretrain_end_time', 'labels', 'epochs', 'test_loss', 'test_accuracy'])
            if ma:
                raise ValueError(ma)
            self.classification = {'optimizer': dct['class_optimizer'], 'loss': dct['class_loss']}
            self.noise = {'base_sigma': dct['gaussian_base_sigma'], 'sigma_factor': dct['gaussian_sigma_factor']}
            self.drop_rate = dct['drop_rate']
            time_str_format = "%y%m%d-%H%M%S"
            self.pretraining = {'start': datetime.strptime(dct['pretrain_start_time'], time_str_format)
                               ,'end':   datetime.strptime(dct['pretrain_end_time'], time_str_format)}
            self.epochs = dct['epochs']
            self.normalization = {'l1': dct['l1'], 'l2': dct['l2']}
            self.accuracy = dct['test_accuracy']
            self.loss = dct['test_loss']
            self.losses = {k: v['loss'] for k,v in dct['labels'].iteritems()}
            self.counts = {k: v['counts'] for k,v in dct['labels'].iteritems()}
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
            self.confusions = {'counts': cc
                              ,'condition_rates':  {k: [float(cc[k][0])/pzd(cc[k][0]+cc[k][1])
                                                       ,float(cc[k][1])/pzd(cc[k][1]+cc[k][0])
                                                       ,float(cc[k][2])/pzd(cc[k][2]+cc[k][3])
                                                       ,float(cc[k][3])/pzd(cc[k][3]+cc[k][2])]
                                                    for k in labels}
                              ,'prediction_rates': {k: [float(cc[k][0])/pzd(cc[k][0]+cc[k][2])
                                                       ,float(cc[k][1])/pzd(cc[k][1]+cc[k][3])
                                                       ,float(cc[k][2])/pzd(cc[k][2]+cc[k][0])
                                                       ,float(cc[k][3])/pzd(cc[k][3]+cc[k][1])]
                                                    for k in labels}
                              }


class EncDecs(AdrGettable):
    def __init__(self, dct, copy_dct = None, nothing = False):
        super(EncDecs, self).__init__(dct, copy_dct, nothing)
        if dct and not copy_dct and not nothing:
            ma = missing_attributes(dct, ['layers', 'encdec_optimizer', 'layer_sizes', 'quality', 'train_quality', 'gaussian_base_sigma', 'gaussian_sigma_factor', 'drop_rate', 'l1', 'l2', 'epochs'])
            if ma:
                raise ValueError(ma)
            self.layers = [v for (_,v) in sorted(dct['layers'].iteritems())]
            self.optimizer = dct['encdec_optimizer']
            self.sizes = dct['layer_sizes']
            self.quality = dct['quality']
            self.train_quality = dct['train_quality']
            self.noise = None \
                         if 'enc_use_noise' in dct and not dct['enc_use_noise'] \
                         else {'base_sigma': dct['gaussian_base_sigma']
                              ,'sigma_factor': dct['gaussian_sigma_factor']}
            self.drop_rate = dct['drop_rate']
            self.normalization = {'l1': dct['l1'], 'l2': dct['l2']}
            self.epochs = dct['epochs']


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

    class Errored(PrintableOf):
        def __init__(self, error_message = ""):
            super(Tester.Errored, self).__init__(error_message)
            self.fail_messages = []
            self.kind = 'errored'

        def bind(self, _):
            return self

        def __repr__(self):
            if not self.fail_messages:
                return super(Tester.Errored, self).__repr__()
            else:
                return "Errored(%s, [%s])" % (self.value, ("\n"+(" "*(8+len(str(self.value))+2))+",").join(self.fail_messages)+"]")

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
            return "Failed([%s])" % (("\n"+(" "*7)+",").join(self.fail_messages))


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


if __name__ == '__main__':
    # Test
    def tw(fun, message = "", eq = None, desired = None):
        def wrapped(value):
            try:
                if eq == None:
                    return Tester.Success(fun(value))
                elif eq:
                    return Tester.Success(value) if fun(value) == desired else Tester.Failed(value, message)
                else:
                    # Well, why not right?
                    return Tester.Success(value) if fun(value) != desired else Tester.Failed(value, message)
            except Exception as e:
                return Tester.Errored(e.message)
        return wrapped

    with open('testcatalog.json', 'r') as f:
        catj = json.load(f)
    et = ['runner-test', 'final-test-batch50']
    tester = Tester(Catalog(catj))
    results = tester.\
        do(0, tw(lambda c: c.get('experiments/'+et[0]+'/results'))).\
        do(1, tw(lambda er: er.get('cross_validation_started'), "Crossvalidation_started wrong", True, True), 0).\
        do(2, tw(lambda er: er.cross_validation_started, "Crossvalidation_started attribute not there", True, True), 0).\
        do(3, tw(lambda c: c.get('experiments/'+et[0]+'settings'))).\
        do(4, tw(lambda es: es.get('drop_rate'), "No get drop_rates", True, [0, 0.1]), 3).\
        do(5, tw(lambda es: es.drop_rate), 3).\
        do(6, tw(lambda es: len(es.gauss_combs), "Length of gauss_combs wrong", True, 5), 3).\
        run()

