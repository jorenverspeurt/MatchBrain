__author__ = 'joren'

import json

from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader
from mindwavemobile.MindwaveDataPoints import *
from signals.primitive import *


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
        #j By default just override the previous measurement...
        if isinstance(datapoint, PoorSignalLevelDataPoint):
            self.building['meta']['noise'] = datapoint.amountOfNoise
            self.building['meta']['contact'] = datapoint.headSetHasContactToSkin()
            if 'noise' in self.unseen:
                self.unseen.remove('noise')
        elif isinstance(datapoint, AttentionDataPoint):
            self.building['eSense']['attention'] = datapoint.attentionValue
            if 'attention' in self.unseen:
                self.unseen.remove('attention')
        elif isinstance(datapoint, MeditationDataPoint):
            self.building['eSense']['meditation'] = datapoint.meditationValue
            if 'meditation' in self.unseen:
                self.unseen.remove('meditation')
        elif isinstance(datapoint, BlinkDataPoint):
            self.building['meta']['blink'] = datapoint.blinkValue
            if 'blink' in self.unseen:
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
            if 'eeg' in self.unseen:
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
        Source.__init__(self, self.get_dict().next)
        self.logger = logging.getLogger('bws')
        self.mindwaveDataPointReader = MindwaveDataPointReader()
        self.mindwaveDataPointReader.start()
        self.finished = False
        self.initialized = False
        self.started = False

    def get_dict(self):
        "Synchronous (!) method to get the next set of brainwave measurements from the sensor"
        while True:
            builder = MeasurementBuilder()
            self.logger.debug('have builder')
            self.finished = False
            while not self.finished:
                datapoint = self.mindwaveDataPointReader.readNextDataPoint()
                self.logger.debug('datapoint read')
                builder.add_measurement(datapoint)
                self.logger.debug('measurement added')
                if builder.is_finished():
                    self.logger.debug('finished building')
                    measurement = builder.get_dict()
                    self.finished = True
                    if not self.initialized:
                        self.logger.debug("Not yet initialized")
                        if not measurement['meta']['noise'] == 25:
                            self.initialized = True
                    else:
                        yield measurement

    def select(self, selector):
        return Transformer([self], {self.getName(): 'd'}, lambda d: d and d[selector])

    def start(self):
        self.started = True
        parent = self
        def internal():
            while parent.started:
                if parent.finished:
                    parent.push()
                    time.sleep(1)
                else:
                    time.sleep(0.1)
        self.t = Thread(target=internal)
        self.t.daemon = True
        self.t.start()

    def stop(self):
        self.started = False
