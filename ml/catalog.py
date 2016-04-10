from __future__ import print_function

import glob
import json
import uuid

import numpy as np


class CatalogManager(object):
    def __init__(self, location):
        self.location = location
        self._catalog_rep = {}
        self._read()

    def _read(self):
        if glob.glob(self.location):
            try: #File may be empty
                with open(self.location, 'r') as f:
                    self._catalog_rep = json.loads(f.read())
            except ValueError:
                self._catalog_rep = {}
        else:
            self._catalog_rep = {}

    def _write(self):
        # Do the dump first in case it throws an error so the file doesn't get screwed...
        dump = json.dumps(self._catalog_rep, indent=2, sort_keys=True, cls=CustomJSONEncoder)
        with open(self.location, 'w') as f:
            f.write(dump)

    def get(self, model_name, param_name = None):
        if param_name is None and model_name in self._catalog_rep:
            return self._catalog_rep[model_name]
        elif param_name and model_name in self._catalog_rep and param_name in self._catalog_rep[model_name]:
            return self._catalog_rep[model_name][param_name]
        else:
            return None

    def set(self, value, model_name, param_name = None):
        if not isinstance(model_name, str):
            print(model_name)
            raise ValueError("model_name must be a string!")
        if param_name is None:
            self._catalog_rep[model_name] = value
        elif model_name and model_name in self._catalog_rep:
            self._catalog_rep[model_name][param_name] = value
        elif isinstance(model_name, str):
            self._catalog_rep[model_name] = { param_name: value }
        self._write()

    def update(self, value, model_name):
        if not isinstance(model_name, str):
            print(model_name)
            raise ValueError("model_name must be a string!")
        if model_name in self._catalog_rep:
            self._catalog_rep[model_name].update(value)
        else:
            self._catalog_rep[model_name] = {}
            self._catalog_rep[model_name].update(value)
        self._write()


class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(CustomJSONEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    class NoIndent(object):
        def __init__(self, value):
            self.value = value

    def no_indent_map(self, dct):
        def handle(v):
            if isinstance(v, (tuple, list)):
                return self.NoIndent(v)
            if isinstance(v, dict):
                return self.no_indent_map(v)
            else:
                return v
        if isinstance(dct, dict):
            return {k: handle(v) for (k,v) in dct.iteritems()}
        elif isinstance(dct, list): # Special case...
            return map(self.no_indent_map, dct)
        else:
            return dct

    def default(self, obj):
        if isinstance(obj, self.NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(obj.value, **self.kwargs)
            return "@@%s@@" % (key,)
        if isinstance(obj, np.ndarray):
            return self.default(self.NoIndent(list(obj)))
        return super(CustomJSONEncoder, self).default(obj)

    def encode(self, obj):
        result = super(CustomJSONEncoder, self).encode(self.no_indent_map(obj))
        for k,v in self._replacement_map.iteritems():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result

