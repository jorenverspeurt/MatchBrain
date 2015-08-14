__author__ = 'joren'

from signals.primitive import *

import datetime as dt
import json
from mindwavemobile.MindwaveDataPoints import *
from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader


class RawMeasurement(dict):
    def __init__(self):
        """

        :type self: RawMeasurement
        """
        super(RawMeasurement, self).__init__(meta={'noise': None, 'contact': None, 'blink': None},
                                             eSense={'attention': None, 'meditation': None},
                                             bands={'delta': None,
                                                    'theta': None,
                                                    'lowAlpha': None,
                                                    'highAlpha': None,
                                                    'lowBeta': None,
                                                    'highBeta': None,
                                                    'lowGamma': None,
                                                    'highGamma': None},
                                             raw=[])


class MeasurementBuilder:
    def __init__(self):
        self.building = RawMeasurement()
        self.done = False
        self.unseen = ['noise', 'attention', 'meditation', 'eeg', 'blink']

    def add_measurement(self, datapoint):
        if isinstance(datapoint, PoorSignalLevelDataPoint):
            self.building['meta']['noise'] = datapoint.amountOfNoise
            self.building['meta']['contact'] = datapoint.headSetHasContactToSkin()
            self.unseen.remove('noise')
        elif isinstance(datapoint, AttentionDataPoint):
            self.building['eSense']['attention'] = datapoint.attentionValue
            self.unseen.remove('attention')
        elif isinstance(datapoint, MeditationDataPoint):
            self.building['eSense']['meditation'] = datapoint.meditationValue
            self.unseen.remove('meditation')
        elif isinstance(datapoint, BlinkDataPoint):
            self.building['meta']['blink'] = datapoint.blinkValue
            self.unseen.remove('blink')
        elif isinstance(datapoint, EEGPowersDataPoint):
            self.building['bands'] = {'delta': datapoint.delta,
                                      'theta': datapoint.theta,
                                      'lowAlpha': datapoint.lowAlpha,
                                      'highAlpha': datapoint.highAlpha,
                                      'lowBeta': datapoint.lowBeta,
                                      'highBeta': datapoint.highBeta,
                                      'lowGamma': datapoint.lowGamma,
                                      'highGamma': datapoint.midGamma}
            self.unseen.remove('eeg')
        elif isinstance(datapoint, RawDataPoint):
            self.building['raw'].append(datapoint._readRawValue())
        else:
            print 'Got a wrongly typed DataPoint object from Mindwave'
        if (not self.unseen) or (self.unseen[0] is 'blink'):
            self.done = True

    def is_finished(self):
        return self.done

    def get_dict(self):
        return self.building

    def __str__(self):
        return json.dumps(self.building, sort_keys=True, indent=2)

    def print_canonical(self):
        canonRep = {'eSense': self.building['eSense'],
                    'eegPower': self.building['bands'],
                    'poorSignalLevel': self.building['meta']['noise']}
        print json.dumps(canonRep, sort_keys=True)


class BrainWaveSource(Source):
    def __init__(self):
        Source.__init__(self, self.get_dict().next, shouldPull= self.has_fresh)
        self.mindwaveDataPointReader = MindwaveDataPointReader()
        self.mindwaveDataPointReader.start()
        self.finished = False
        self.history = {}
    def get_dict(self):
        "Synchronous (!) method to get the next set of brainwave measurements from the sensor"
        while True:
            builder = MeasurementBuilder()
            self.finished = False
            while not self.finished:
                datapoint = self.mindwaveDataPointReader.readNextDataPoint()
                builder.add_measurement(datapoint)
                if builder.is_finished():
                    builder.print_canonical()
                    self.record(builder.get_dict())
                    self.finished = True
                    yield builder.get_dict()

    def has_fresh(self):
        return self.finished

    def record(self, value):
        self.history[str(dt.datetime.now())] = value

    def select(self, selector):
        return Transformer([self], {self.getName(): 'd'}, lambda d: d and d[selector])