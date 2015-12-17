from __future__ import print_function

import glob
import json
import random
from datetime import datetime, timedelta
from itertools import repeat, takewhile, groupby

from signals.primitive import GenSource, SignalBlock, Transformer

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
                takewhile(lambda c: c!='2', fname)
            #tester_names = sorted(list(set(map(tname_for, all_dict.keys))))
            all_dict = reduce(dict.update,
                              {f: all_dict[max(g)]
                               for f,g in groupby(sorted(all_dict.iterkeys()), tname_for)}, {})
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


