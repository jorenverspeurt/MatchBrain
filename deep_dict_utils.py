def get(key):
    if isinstance(key, list):
        return (lambda x: [x[k] for k in key])
    else:
        return (lambda x: x[key])

def put(key, value):
    if isinstance(key, list):
        return (lambda x: dict(x, **{key[i]: value[i] for i in xrange(len(key))}))
    else:
        return (lambda x: dict(x, **{key: value}))
    
def only(keys):
    return (lambda x: {k: v for k,v in x.iteritems() if k in keys})

def drop(key):
    if isinstance(key, list):
        return (lambda x: {k: v for k,v in x.iteritems() if not k in key})
    else:
        return (lambda x: {k: v for k,v in x.iteritems() if k != key})

def on(key, fun):
    return (lambda x: put(key, fun(x[key]))(x))

def over(key, fun):
    return (lambda x: put(key, dmap(fun, x[key]))(x) 
                      if isinstance(x[key], dict) else (
                          put(key, map(fun, x[key]))(x)
                          if isinstance(x[key], list) else on(key, fun)(x)))

def chain(*funs):
    return (lambda x: reduce(lambda p,c: c(p), funs, x))

def liftup(key):
    return (lambda x: dict(drop(key)(x), **{str(key)+"_"+str(k): v for k,v in x[key].iteritems()}))

def rec_liftup(key):
    return (lambda x: dict(drop(key)(x), **reduce(lambda p, c: rec_liftup(c)(p) if isinstance(p[c], dict) else p,
                                                  liftup(key)({key: x[key]}).iterkeys(),
                                                  liftup(key)({key: x[key]}))))

def combine(sources, fun, destination, keep = False):
    dest_less_srcs = sources+[]
    if destination in dest_less_srcs:
        dest_less_srcs.remove(destination)
    return (lambda x: chain(put(destination, 
                                fun(*get(sources)(x))), 
                            drop(dest_less_srcs if not keep else []))(x))

def nmap(fun, coll):
    return map(lambda e: map(fun, e), coll)

def dmap(fun, dct = None):
    if not dct is None:
        return { k: fun(v) for k,v in dct.iteritems() }
    else:
        return (lambda x: dmap(fun, x))

def keymap(fun, dct = None):
    if not dct is None:
        return { fun(k): v for k,v in dct.iteritems() }
    else:
        return (lambda x: keymap(fun, x))

def smart_combine(old, new):
    if isinstance(old, list):
        if isinstance(new, list):
            return old + new
        else:
            return old + [new]
    elif isinstance(old, dict):
        if not all(ok in new.keys() for ok in old.iterkeys()):
            return old
        return { k: smart_combine(old[k], new[k]) 
                 for k in old.iterkeys() }
    else:
        return [old, new]

def columnify(old_keys = [], type_col = "", value_col = "", match_length = True):
    tc = type_col or "type"
    vc = value_col or "value"
    def internal(dct):
        ok = old_keys or list(sorted(dct.iterkeys()))
        if match_length:
            vals = get(ok)(dct)
            lens = map(lambda e: len(e) if isinstance(e, list) or isinstance(e, dict) else 1, vals)
        else:
            lens = [1 for _ in ok]
        res = combine(ok, lambda *ov: reduce(smart_combine, ov), vc)(dct)
        res[tc] = [ok[i] for i in xrange(len(ok)) for _ in xrange(lens[i])]
        return res
    return internal

def flip(dct):
    assert all(isinstance(v, dict) for v in dct.itervalues())
    return {ik: {ok: dct[ok][ik] for ok in dct.iterkeys()} for ik in dct.values()[0].iterkeys()}

def detvalues(d):
    return [d[k] for k in sorted(d.iterkeys())]

def trace(x):
    print(x)
    return x

def iden(x):
    return x
