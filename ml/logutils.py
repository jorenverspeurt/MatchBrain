from __future__ import print_function

import glob
import json
from datetime import datetime, timedelta
from itertools import repeat
from random import sample

from core.TrainView import phase_names
from signals.primitive import GenSource, SignalBlock, Transformer


def loadfile(name):
    with open(name) as f:
        data = json.loads(f.read())
    return data

def dumpfile(name, data):
    with open(name, 'r') as f:
        f.write(json.dumps(data))

def loadall():
    all_files = {}
    for f in glob.glob("/home/joren/PycharmProjects/MatchBrain/logs/*.json"):
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

def get_phase():
    pass

class LogSourceMaker(object):
    def __init__(self, clean_seconds = 3):
        all_dict = loadall()
        self.phases = phase_names
        #self.phases.append('none')
        self.raws_per_phase = []
        cur_phase = phase_names[0]
        for p in all_dict.keys():
            cur_part = all_dict[p]
            seen_phase = False
            clean_counter = clean_seconds
            for li in cur_part:
                if "data" in li:
                    if "train" in li["data"]:
                        cur_phase = li["data"]["train"]["phase"]
                        seen_phase = True
                        clean_counter = clean_seconds
                    elif "brainwave" in li["data"]:
                        if clean_counter > 0:
                            clean_counter -= 1
                        else:
                            self.raws_per_phase.append((cur_phase
                                                        if seen_phase
                                                        else "DISTRACT"
                                                       ,li["data"]["brainwave"]["raw"]))
        #TODO ?

    def get_block(self):
        self.ph_source = GenSource(li[0] for rep in repeat(self.raws_per_phase) for li in sample(rep,len(rep)))
        self.bw_source = GenSource(li[1] for rep in repeat(self.raws_per_phase) for li in sample(rep,len(rep)))
        result = SignalBlock(
            [self.bw_source, self.ph_source],
            [Transformer(
                [self.bw_source, self.ph_source],
                {self.bw_source.getName(): 'b', self.ph_source.getName(): 'd'},
                lambda b,d: (b,d))
            ]
        )
        self.bw_source.callback = result.stop
        self.ph_source.callback = result.stop
        #result.sinks.append(Sink([self.bw_source, self.ph_source], lambda x: print(x)))
        return result


