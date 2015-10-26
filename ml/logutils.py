import json
import glob
from datetime import datetime, timedelta

from signals.primitive import Source
from core.TrainView import phase_names


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
        if (prev_ms > ms or \
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


class LogSource(Source):
    def __init__(self):
        Source.__init__(self, self.nextvalue)
        self.log_dict = loadall()
        self.participant = self.log_dict.keys()[0] #Some "random" participant
        self.phases = phase_names
        self.phase = get_phase(self.log_dict[phase_names[0]],)#TODO implement
        self.index = 0

    def nextvalue(self):
        pass #TODO implement

