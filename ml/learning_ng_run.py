import cPickle
import glob
import gzip
import os

from learning_ng import PretrainedClassifier as Classifier, safe_head

default_data_location = safe_head(glob.glob(os.path.join(os.path.dirname(__file__), 'normalized.pkl.gz')))

class LearningRunner(object):
    def __init__(self
                # Runner-general
                ,data_location = None
                ,test_run = False
                ,cross_val = True
                # Classifier-general
                ,epochs = 300000
                ,batch_size = 50
                ,max_layer_sizes = 0
                ,encdecs_name = ""
                ,model_name = ""
                # Classifier-specific
                ,encdec_optimizers = ('adadelta',)
                ,class_optimizers = ('adadelta',)
                ,class_losses = ('categorical_crossentropy',)
                ,drop_rates = (0.0,)
                ,gauss_base_sigmas = (0.0,)
                ,gauss_sigma_factors = (1.0,)
                ,l1s = (0.0,)
                ,l2s = (0.0,) ):
        self.data_location = data_location or default_data_location
        with gzip.open(data_location, 'rb') as f:
            self.data = cPickle.load(f)
        if cross_val:
            self.cross_val_drops = list(self.data.keys()) if cross_val else []
        else:
            self.cross_val_drops = []
        self.cross_val_index = 0
        self.cross_val_test_data = None
        if test_run:
            self.epochs = 1
            self.batch_size = 1000
        else:
            self.epochs = epochs
            self.batch_size = batch_size
        self.max_layer_sizes = max_layer_sizes
        self.encdecs_name = encdecs_name
        self.model_name = model_name
        self.encdec_optimizers = encdec_optimizers
        self.class_optimizers = class_optimizers
        self.class_losses = class_losses
        self.drop_rates = drop_rates
        self.gauss_base_sigmas = gauss_base_sigmas
        self.gauss_sigma_factors = gauss_sigma_factors
        self.l1s = l1s
        self.l2s = l2s

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
                                current_name = self.model_name + ("-%s%s%s%s" % (j,k,l,m)) + ifxval
                                enc_name = self.encdecs_name + ("-%s%s%s%s%s" % (i,j,k,l,m)) + ifxval
                                pc = Classifier(
                                    data,
                                    labels,
                                    self.batch_size,
                                    self.epochs,
                                    current_name,
                                    "",
                                    enc_name,
                                    eo,
                                    None,
                                    None,
                                    dr,
                                    gbs,
                                    gsf,
                                    l1,
                                    l2
                                )
                                pc.new_encdecs(True, True, True)
                                pc.pretrain()
                                pc.cap_data()
                                for (n,co) in enumerate(self.class_optimizers):
                                    for (o,cl) in enumerate(self.class_losses):
                                        suffix = "-%s%s%s%s%s%s" % (j,k,l,m,n,o)
                                        current_name = self.model_name + suffix + ifxval
                                        pc.catalog_manager.update({
                                            'cross_validation': {
                                                'enabled': bool(self.cross_val_drops),
                                                'drops': self.cross_val_drops,
                                                'index_reached': self.cross_val_index
                                            },
                                            suffix: {
                                                'finished': False
                                            }
                                        }, self.model_name)
                                        pc.model_name = current_name
                                        pc.cls_opt = co
                                        pc.cls_lss = cl
                                        pc.new_model(False, True)
                                        history = pc.finetune()
                                        with gzip.open(os.path.join(os.path.dirname(default_data_location), current_name + '.history.pkl.gz'), 'wb') as f:
                                            cPickle.dump(history, f, 2)
                                        pc.catalog_manager.set({ suffix: { 'finished': True, 'test_accuracy': pc._model_info['test_accuracy'] } })
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
    runner = LearningRunner(
        data_location='./normalized_data.pkl.gz',
        test_run=False,
        cross_val = True,
        # Classifier-general
        epochs = 300000,
        batch_size = 50,
        max_layer_sizes = 0,
        encdecs_name = "",
        model_name = "",
        # Classifier-specific,
        encdec_optimizers = ('adadelta',),
        class_optimizers = ('adadelta',),
        class_losses = ('categorical_crossentropy',),
        drop_rates = (0.0,),
        gauss_base_sigmas = (0.0,),
        gauss_sigma_factors = (1.0,),
        l1s = (0.0,),
        l2s = (0.0,)
    )


