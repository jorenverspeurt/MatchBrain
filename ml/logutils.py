from __future__ import print_function

import glob
import json
import random
from datetime import datetime, timedelta
from itertools import repeat, takewhile, groupby
import numpy as np
import gzip, cPickle

from signals.primitive import GenSource, SignalBlock, Transformer
from processing import *

phase_names = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']

def interp_vals(val1, val2, index, total):
    return val1 + ( (val2-val1) * (index/total) )

def interp_ls(l1, l2, i):
    return l1[i:] + l2[0:i]

def loadfile(name):
    with open(name) as f:
        data = json.loads(f.read())
    return data

def dumpfile(name, data):
    with open(name, 'r') as f:
        f.write(json.dumps(data))

def loadall(logfolder="/home/joren/PycharmProjects/MatchBrain/logs/"):
    all_files = {}
    for f in glob.glob(logfolder+"*.json"):
        all_files[f] = loadfile(f)
    return all_files

def entrytypes(data):
    return set(map(lambda a: tuple(a.keys())
                  ,data))

def allentrytypes():
    all_data = loadall()
    return {k[0:-5]: entrytypes(all_data[k]) for k in all_data.iterkeys()}

def msecs_to_asctime(data):
    header, rest = data[0], data[1:]
    cur_time = datetime.strptime(header['startTime'], '%Y-%m-%dT%H:%M')
    one_s = timedelta(seconds=1)
    result = [header]
    prev_ms = 0
    prev_type = ''
    for entry in rest:
        d_o = {'data': entry['data']}
        ms = entry['msecs']
        cur_type = d_o['data'].keys()[0]
        if (prev_ms > ms or
            not (prev_type == 'brainwave' and
                 cur_type != 'brainwave')
            ):
            cur_time += one_s
        delta = timedelta(microseconds=ms*1000)
        d_o['asctime'] = (cur_time + delta).isoformat(' ').replace('.',',')
        result.append(d_o)
        prev_ms = ms
        prev_type = cur_type
    return result

def all_m_to_a():
    all_data = loadall()
    with_msecs = {k:v for (k,v) in all_data.iteritems() if 'msecs' in v[1]}
    for name in with_msecs:
        without = msecs_to_asctime(with_msecs[name])
        with open(name[0:-5]+"-asctime.json", 'w') as new_file:
            new_file.write(json.dumps(without))
    return with_msecs.keys()

class LogSourceMaker(object):
    def __init__(self, clean_seconds = 3, logfolder = None, phases = None, cross_val = False):
        if logfolder:
            all_dict = loadall(logfolder=logfolder)
        else:
            all_dict = loadall()
        self.seen_val_names = []
        if cross_val:
            def tname_for(fname):
                just_f = fname.split('/')[-1]
                return just_f.split('2015')[0]
            #tester_names = sorted(list(set(map(tname_for, all_dict.iterkeys()))))
            #print(tester_names)
            #print({f: max(g) for f,g in groupby(sorted(all_dict.iterkeys()), tname_for)})
            all_dict = {f: all_dict[max(g)]
                        for f,g in groupby(sorted(all_dict.iterkeys()), tname_for)}
            val_name = random.choice(all_dict.keys())
            self.seen_val_names.append(val_name)
            self.val_dict = {val_name: all_dict[val_name]}
            self.all_dict = {k:v for (k,v) in all_dict.iteritems() if not k is val_name}
        else:
            self.val_dict = {}
            self.all_dict = all_dict
        self.phases = phases or phase_names
        #self.phases.append('none')
        self.clean_seconds = clean_seconds
        self.raws_per_phase = self.to_raws_per_phase(all_dict)
        self.rpp_val = self.to_raws_per_phase(self.val_dict)

    def cross_val_next(self):
        all_dict = self.all_dict
        all_dict.update(self.val_dict)
        next_val = random.choice(all_dict.keys())
        while next_val in self.seen_val_names:
            next_val = random.choice(all_dict.keys())
        self.val_dict = {next_val: all_dict[next_val]}
        self.all_dict = {k:v for (k,v) in all_dict.iteritems() if not k is next_val}
        self.raws_per_phase = self.to_raws_per_phase(self.all_dict)
        self.rpp_val = self.to_raws_per_phase(self.val_dict)

    def to_raws_per_phase(self, a_dict):
        result = []
        cur_phase = phase_names[0]
        for p in a_dict.keys():
            cur_part = a_dict[p]
            seen_phase = False
            clean_counter = self.clean_seconds
            for li in cur_part:
                if "data" in li:
                    if "train" in li["data"] and li["data"]["train"]["phase"] in self.phases:
                        cur_phase = li["data"]["train"]["phase"]
                        seen_phase = True
                        clean_counter = self.clean_seconds
                    elif "brainwave" in li["data"]:
                        if clean_counter > 0:
                            clean_counter -= 1
                        else:
                            result.append((cur_phase
                                           if seen_phase
                                           else "DISTRACT"
                                          ,li["data"]["brainwave"]["raw"]))
                    else:
                        # Skip data for "phases" that aren't in the list
                        pass
        return result

    def get_block(self, shift = 1):
        a_meas = self.raws_per_phase[0][1]
        self.ph_source = GenSource(li[0]
                                   for rep in repeat(self.raws_per_phase, len(a_meas))
                                   for li in rep)
        self.bw_source = GenSource(li
                                   for i in xrange(0, len(a_meas), shift)
                                   for li in map(lambda t: interp_ls(t[0][1], t[1][1], i)
                                                ,zip(self.raws_per_phase[0:-1], self.raws_per_phase[1:])))
        block = SignalBlock(
            [self.bw_source, self.ph_source],
            [Transformer(
                [self.bw_source, self.ph_source],
                {self.bw_source.getName(): 'b', self.ph_source.getName(): 'd'},
                lambda b,d: (b,d))
            ]
        )
        self.bw_source.callback = block.stop
        self.ph_source.callback = block.stop
        #result.sinks.append(Sink([self.bw_source, self.ph_source], lambda x: print(x)))
        return block

class NormSourceMaker(object):
    def __init__(self, datafolder = None, phases = None, cross_val = False):
        self.data = None
        if datafolder:
            with gzip.open(datafolder+"normalized.pkl.gz",'rb') as f:
                self.data = cPickle.load(f)
        else:
            paths = glob.glob("~/PycharmProjects/MatchBrain/ml/normalized.pkl.gz")
            if len(paths) > 0:
                with gzip.open(paths[0]) as f:
                    self.data = cPickle.load(f)
        if not self.data:
            return
        self.phases = phases or phase_names
        self.cross_val_keys = list(self.data.keys())
        self.cross_val_index = 0 if cross_val else None

    def cross_val_next(self):
        self.cross_val_index = (self.cross_val_index + 1) % len(self.cross_val_keys)

    def get_block(self):
        self.source = GenSource(e
                                for i in xrange(len(self.cross_val_keys))
                                if not i == self.cross_val_index
                                for e in self.data[self.cross_val_keys[i]])
        self.ph_source = Transformer([self.source], {self.source.getName(): 'd'}, lambda d: d['phase'])
        self.bw_source = Transformer([self.source], {self.source.getName(): 'd'}, lambda d: d['raw'])
        block = SignalBlock(
            [self.source],
            [self.bw_source, self.ph_source]
        )
        self.source.callback = block.stop
        return block


if __name__ == '__main__':
    """
    Process all current logs into a single file
    Keep only needed data, perform preprocessing transforms
    Compute necessary statistics
    """
    drop_seconds = 4
    shift = 4 # TODO: actually shift...
    ###
    # type DEntry = { "brainwave": { "eSense": {"meditation": int, "attention": int}
    #                                          , "raw": [int]
    #                                          , "meta": {"blink": bool, "noise": int, "contact": bool}
    #                                          , "bands": { "lowGamma": int
    #                                                     , "highAlpha": int
    #                                                     , "highGamma": int
    #                                                     , "lowAlpha": int
    #                                                     , "delta": int
    #                                                     , "theta": int
    #                                                     , "lowBeta": int
    #                                                     , "highBeta": int}}}
    #             | { "train": {"phase": phase} } where phase in Phases
    #             | { "game": gstate or gobjective or gscore}
    #               where gstate = {"state": state} where state in GameModel.STATES
    #                     gobjectives = {"objective": {"duration": int, obj: {...}, "type": obj}} where obj in Objectives
    #                     gscore = {"score": int}
    #             | { "event": mouse or key }
    #               where mouse = {"mouse": drag or press}
    #                             where drag = {"drag": {"y": int, "x": int, "buttons": int, "dx": int, "dy": int}}}
    #                                   press = {"press": {"y": int, "x": int, "buttons": int}}
    #                     keyboard = {"key": symbol} where symbol = string
    # type Nick = {"nick": str, "version": float, "startTime": str}
    # all_logs :: {filename: [Nick or {"data": DEntry, ascTime: str}]}
    all_logs = loadall()
    np.set_printoptions(precision=3, suppress=True)

    # handle_data_entry :: ({"data": DEntry} -> bw or tr or None) or ({ other } -> None)
    #                      where bw = { "type": "brainwave", "raw": [int], "eSense": ...}
    #                            tr = { "type": "phase", "phase": phase }
    def handle_data_entry(de):
        if "data" in de:
            data = de["data"]
            if "brainwave" in data \
                    and sum(data["brainwave"]["eSense"].itervalues()) != 0\
                    and data["brainwave"]["meta"]["contact"]:
                bw = data["brainwave"]
                return { "type": "brainwave"
                       , "raw": bw["raw"]
                       , "eSense": bw["eSense"]
                       , "bands": bw["bands"]}
            elif "train" in data:
                return { "type": "phase"
                       , "phase" : data["train"]["phase"] }
            else:
                return None # Include others as needed
        else:
            return None

    # (filter(None, ...) removes Nones and other falsies)
    result = { fname: filter(None, map(handle_data_entry, fc))
               for (fname, fc) in all_logs.iteritems() }
    # result: { filename: [bw or tr] }

    # change_drop :: (str, int, [bw]) -> (bw or tr) -> (str, int, [bw])
    def change_drop((cur_phase, count, data_acc), new_data):
        if count > 0:
            return (cur_phase, count - 1, data_acc)
        elif new_data["type"] == "phase":
            if new_data["phase"] == "NEWGAME":
                return ("NEWGAME", 1e10, data_acc) # Ignore newgame data for now
            else:
                return (new_data["phase"], drop_seconds, data_acc)
        elif new_data["type"] == "brainwave":
            # Ugh, imperative rubbish, but hey, it saves some memory I guess
            new_data["phase"] = cur_phase
            data_acc.append(new_data)
            return (cur_phase, count, data_acc)
        else:
            # Let's find out which case I'm not covering here...
            # Should be none for now
            print(new_data)

    # Go over the data, dropping what needs to be dropped
    # result :: { filename: [bw] }
    result = { fname: reduce(change_drop, fc, (phase_names[0], drop_seconds, []))[2]
               for (fname, fc) in result.iteritems() }

    # A convoluted way of mapping fix_length over just result['raw'] and leaving the rest untouched
    # Replace raw data by preprocess'ed version
    # result :: { filename: [ {"type": "brainwave", "raw": [array], eSense: {...}, bands: {...}}
    result = { fname: map(lambda r: dict(r, **{ 'raw': merge(*map(lambda f: f(fix_length(r['raw'], 512))
                                                                 ,[wavelet_trans, fourier_trans, extremes])) })
                         ,fc)
               for (fname, fc) in result.iteritems() }

    # Needed stats for normalization: mean and standard deviation
    # statd :: array -> {"mean": array, "std": array, 'n': int}
    statd = lambda arr: {
        'mean': np.mean(arr, axis = 0),
        'std': np.std(arr, axis = 0),
        'n': len(arr)
    }
    # A way to get the values out of a dictionary sorted by their key name alphabetically (so it's deterministic)
    # detvalues :: dict -> list
    detvalues = lambda d: [d[k] for k in sorted(d.iterkeys())]
    # Ways of coercing collections of integer brainwave records into lists
    nfs = [('raw', list), ('bands', detvalues), ('eSense', detvalues)]

    # getstats :: {"raw" or "bands" or "eSense" : ... } -> [("raw", ...->list) or ("bands", ...) or ...]
    # -> {"raw" or ... : {"mean": array, "std": array, "n": int}}
    def getstats(res, l_cat_f):
        return { cat: statd([f(e[cat])
                             for (fname, fc) in res.iteritems()
                             for e in fc])
                 for (cat, f) in l_cat_f }

    allstats = getstats(result, nfs)
    # Group sessions per player
    # Keep both the original preproc data and averages in the new result
    pname_for = lambda f: ''.join(takewhile(lambda c: c!='2', f)).split('/')[-1]
    pnames = { pname_for(fname) for fname in result.iterkeys() }
    perplayer = { pname: sum(((fc if isinstance(fc,list) else [fc])
                             for (fname, fc) in result.iteritems()
                             if pname_for(fname) == pname), [])
                  for pname in pnames }
    perplayer = { pname: dict({ 'stats': getstats({ pname: pc }, nfs)}
                             ,**{ fname: { 'data': fc
                                         , 'stats': getstats({ fname: fc }, nfs) }
                                  for (fname, fc) in result.iteritems()
                                  if pname_for(fname) == pname})
                  for (pname, pc) in perplayer.iteritems() }
    # result :: { 'stats': { ... }
    #           , 'players': { pname1: { 'stats': { 'raw': { ... }, ... }
    #                                  , f_p1: { 'stats': { ... }, 'data': [...] }
    #                        , ... }
    #           }
    result = { 'stats': allstats, 'players': perplayer }
    with gzip.open('unscaled.pkl.gz','wb') as f:
        cPickle.dump(result, f, 2)
    def dict_without(dic, keys):
        return { k: v if not isinstance(v, dict) else dict_without(v, keys) for (k, v) in dic.iteritems() if not k in keys }
    def normalized_by(scaled, scaling):
        def dict_aware_norm(value, mean, std):
            if isinstance(value, dict):
                return { k: (v-mean)/std for (k,v) in value.iteritems() }
            else:
                return (value-mean)/std

        return [ { k: dict_aware_norm(v, scaling[k]['mean'], scaling[k]['std']) if k in scaling else v
                   for (k,v) in e.iteritems()
                   if not k == 'type' } #Useless to keep this, it's all 'brainwave' at this point
                 for e in scaled ]

    scaled = { pname: normalized_by(fe['data'], result['stats'])
               for (pname, pe) in result['players'].iteritems()
               for (fname, fe) in pe.iteritems()
               if fname != 'stats' }
    with gzip.open('normalized.pkl.gz','wb') as f:
        cPickle.dump(scaled, f, 2)

