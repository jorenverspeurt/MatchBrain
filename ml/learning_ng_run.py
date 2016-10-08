import cPickle
import glob
import gzip
import os, sys
import json

from catalog import CatalogManager, DummyCatalogManager, named_catalog
from learning_ng import PretrainedClassifier as Classifier, safe_head, time_str

default_data_location = safe_head(glob.glob(os.path.join(os.path.dirname(__file__), 'normalized.pkl.gz')))

class LearningRunner(object):
    def __init__(self
                # Runner-general
                ,data_location = './normalized.pkl.gz'
                ,test_run = False
                ,cross_val = True
                # Classifier-general
                ,epochs = 1000
                ,batch_size = 100
                ,max_layer_sizes = 0
                ,encdecs_name = ""
                ,model_name = ""
                # Classifier-specific
                ,encdec_optimizers = ('rmsprop',)
                ,class_optimizers = ('adadelta',)
                ,class_losses = ('categorical_crossentropy',)
                ,drop_rates = (0.0,)
                ,gauss_base_sigmas = (0.0,)
                ,gauss_sigma_factors = (1.0,)
                ,l1s = (0.0,)
                ,l2s = (0.0,)
                ,enc_use_drop = False
                ,enc_use_noise = True
                ,mod_use_drop = True
                ,mod_use_noise = False
                ,catalog_name = ""
                ,starting_index = 0):
        self.data_location = data_location or default_data_location
        with gzip.open(data_location, 'rb') as f:
            self.data = cPickle.load(f)
        if cross_val:
            self.cross_val_drops = list(self.data.keys()) if cross_val else []
        else:
            self.cross_val_drops = []
        self.cross_val_index = starting_index if cross_val else -1
        self.cross_val_test_data = None
        if test_run:
            self.epochs = 4
            self.batch_size = 1000
        else:
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
        self.max_layer_sizes = int(max_layer_sizes)
        self.encdecs_name = str(encdecs_name) or str(model_name) or time_str()
        self.model_name = str(model_name) or str(encdecs_name) or time_str()
        self.encdec_optimizers = map(str, encdec_optimizers)
        self.class_optimizers = map(str, class_optimizers)
        self.class_losses = map(str, class_losses)
        self.drop_rates = map(float, drop_rates)
        self.gauss_base_sigmas = map(float, gauss_base_sigmas)
        self.gauss_sigma_factors = map(float, gauss_sigma_factors)
        self.l1s = map(float, l1s)
        self.l2s = map(float, l2s)
        self.enc_use_drop = bool(enc_use_drop)
        self.enc_use_noise = bool(enc_use_noise)
        self.mod_use_drop = bool(mod_use_drop)
        self.mod_use_noise = bool(mod_use_noise)
        self.cat = DummyCatalogManager if test_run else (named_catalog(catalog_name) if catalog_name else CatalogManager)

    def run(self):
        while self.cross_val_index < len(self.cross_val_drops):
            if self.cross_val_drops:
                labels, data = self.split_data(True)
            else:
                labels, data = self.split_data(False)
            gauss_combs = [(0, 1)] if 0 in self.gauss_base_sigmas else []
            gauss_combs += [(gbs, gsf)
                            for gbs in self.gauss_base_sigmas
                            for gsf in self.gauss_sigma_factors
                            if gbs != 0]
            for (i,eo) in enumerate(self.encdec_optimizers):
                for (j,dr) in enumerate(self.drop_rates):
                    for (k,(gbs,gsf)) in enumerate(gauss_combs):
                        for (l, l1) in enumerate(self.l1s):
                            for (m, l2) in enumerate(self.l2s):
                                ifxval = ("-x%s" % self.cross_val_index if self.cross_val_drops else "")
                                current_name = self.model_name + ifxval + ("-%s%s%s%s" % (j,k,l,m))
                                enc_name = self.encdecs_name + ifxval + ("-%s%s%s%s%s" % (i,j,k,l,m))
                                pc = Classifier(
                                    data = data,
                                    labels = labels,
                                    batch_size = self.batch_size,
                                    epochs = self.epochs,
                                    model_name = current_name,
                                    model_dir = "",
                                    encdecs_name = enc_name,
                                    encdec_optimizer = eo,
                                    # Not used for now, set later
                                    class_optimizer = None,
                                    # idem
                                    class_loss = None,
                                    drop_rate = dr,
                                    gauss_base_sigma = gbs,
                                    gauss_sigma_factor = gsf,
                                    l1 = l1,
                                    l2 = l2,
                                    catalog_class = self.cat
                                )
                                # TODO do something with max-layer-sizes here
                                pc.new_encdecs(compile = True,
                                               use_dropout = self.enc_use_drop,
                                               use_noise = self.enc_use_noise)
                                pc.pretrain()
                                pc.cap_data()
                                for (n,co) in enumerate(self.class_optimizers):
                                    for (o,cl) in enumerate(self.class_losses):
                                        suffix = "-%s%s%s%s%s%s%s" % (i,j,k,l,m,n,o)
                                        current_name = self.model_name + ifxval + suffix
                                        start_time = time_str()
                                        pc.catalog_manager.update({
                                            'cross_validation': {
                                                'enabled': bool(self.cross_val_drops),
                                                'drops': self.cross_val_drops,
                                                'index_reached': self.cross_val_index
                                            },
                                            'settings': {
                                                '0-eo': self.encdec_optimizers,
                                                '1-dr': self.drop_rates,
                                                '2-(gbs,gsf)': gauss_combs,
                                                '3-l1s': self.l1s,
                                                '4-l2s': self.l2s,
                                                '5-co': self.class_optimizers,
                                                '6-cl': self.class_losses
                                            },
                                            ifxval + suffix: {
                                                'finished': False,
                                                'start_time': start_time
                                            }
                                        }, self.model_name)
                                        pc.model_name = current_name
                                        pc.cls_opt = co
                                        pc.cls_lss = cl
                                        pc.new_model(fresh = False,
                                                     compile = True,
                                                     use_dropout = self.mod_use_drop,
                                                     use_noise = self.mod_use_noise)
                                        history = pc.finetune()
                                        with gzip.open(os.path.join(os.path.dirname(default_data_location), current_name + '.history.pkl.gz'), 'wb') as f:
                                            cPickle.dump(history, f, 2)
                                        pc.catalog_manager.set({ 'finished': True,
                                                                 'test_accuracy': pc._model_info['test_accuracy'],
                                                                 'start_time': start_time,
                                                                 'end_time': time_str() }, self.model_name, ifxval + suffix)
            self.cross_val_index += 1

    def split_data(self, xval = False, label_sel = 'phase', data_sel = 'raw'):
        left_out = self.cross_val_drops[self.cross_val_index] if xval else None
        unsplit = [(e[label_sel], e[data_sel])
                   for name in self.data.iterkeys()
                   if not name == left_out
                   for e in self.data[name]]
        if xval:
            self.cross_val_test_data = [(e[label_sel], e[data_sel]) for e in self.data[left_out]]
        return zip(*unsplit)

if __name__ == '__main__':
    runner = None
    if len(sys.argv) <= 1:
        runner = LearningRunner(
            data_location='./normalized.pkl.gz',
            test_run = False,
            cross_val = True,
            # Classifier-general
            epochs = 500,
            batch_size = 100,
            max_layer_sizes = 0,
            encdecs_name = "final-test-short-2",
            model_name = "final-test-short-2",
            # Classifier-specific,
            encdec_optimizers = ('rmsprop',),
            class_optimizers = ('adadelta',),
            class_losses = ('categorical_crossentropy',),
            drop_rates = (0.10,),
            gauss_base_sigmas = (0.10,),
            gauss_sigma_factors = (1.0,),
            l1s = (0.0,),
            l2s = (0.0,),
            catalog_name = "short-2"
        )
        print("Default runner made")
    else:
        try:
            location = safe_head(glob.glob(os.path.join(os.path.dirname(__file__), sys.argv[1])))
            print("Loading settings from "+location)
            with open(location, 'r') as f:
                settings = json.loads(f.read())
            the_name = location.split('/')[-1].replace('.settings', '').replace('.json', '')
            if (not "encdecs_name" in settings) and (not "model_name" in settings):
                settings["encdecs_name"] = "final-test-"+the_name
                settings["model_name"] = "final-test-"+the_name
            if (not "catalog_name" in settings) or ("catalog_name" in settings and not settings["catalog_name"]):
                settings["catalog_name"] = the_name+".json"
            runner = LearningRunner(**{str(k): v for k,v in settings.iteritems()})
            print("Runner made")
        except Exception as e:
            print(e)
    if not runner is None:
        runner.run()


