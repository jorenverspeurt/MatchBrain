__author__ = 'joren'

import logging
import random
import threading
import time
from datetime import datetime as dt

import pyglet.resource
from cocos import director
from cocos.layer import MultiplexLayer
from cocos.scene import Scene

from LogHelpers import CustomHandler
from Menus import MainMenu
from ml.learning import AutoTransformer
from ml.processing import Preprocessing
from signals.mindwave import BrainWaveSource
from signals.primitive import Sink, Source

version = 0.2

def setup(test=False, phase_source = None, train_end_calls = []):
    director.director.init(width=1056, height=700, caption='MatchBrain')
    scene = Scene()
    scene.add(MultiplexLayer(
        MainMenu(test, phase_source, train_end_calls),
    ), z = 1)
    return scene

def game(scene):
    director.director.run(scene)

def cleanup():
    pass

if __name__ == '__main__':
    # (Data) logging stuff
    nickname = raw_input('Enter nickname: ')
    test = nickname == 'test'
    demo = nickname == 'demo'
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    rootHandler = logging.StreamHandler()
    rootLogger.addHandler(rootHandler)
    ph_source = Source(lambda: "DISTRACT")
    if not demo:
        if not test:
            dataLogger = logging.getLogger('data')
            dataLogger.setLevel(logging.INFO)
            jsonHandler = CustomHandler(
                '../logs/'+nickname+dt.now().strftime('%Y-%m-%dT%H:%M')+'.json',
                {'nick': nickname, 'startTime': dt.now(), 'version': version})
            dataLogger.addHandler(jsonHandler)
            bwSource = BrainWaveSource()
        else:
            class LogMock(object):
                def debug(self, string):
                    pass
                def info(self, string):
                    print(string)
                def log(self, string, level):
                    print(string)
            dataLogger = LogMock()
            bwSource = Source(lambda: [random.random()*4096-2048 for i in xrange(512)])
        # prep = Preprocessing(bwSource)
        # ae = AutoTransformer(prep.output_dim, prep, ph_source, batch_size=1, num_sizes=5, epochs=15, model_name=nickname)
        # ae.load_encdecs("o-adadelta-dr-0.001-gbs-0.001-gsf-2-l1-0-l2-0.001")
        bwSink = Sink([bwSource], logging.getLogger('data.brainwave').info)
        def bwUpdate(sources, sink, seconds):
            dataLogger.debug('before internal')
            def internal():
                #next_call = time.time()
                while True:
                    dataLogger.debug('internal 1')
                    #next_call = next_call + seconds
                    for source in sources:
                        source.push()
                    dataLogger.debug('internal pushed')
                    sink.pull()
                    dataLogger.debug('internal pulled')
                    time.sleep(seconds) #abs(next_call - time.time())) #DIRTY
            return internal
        timer = threading.Thread(target = bwUpdate([bwSource, ph_source], bwSink, 1))
        timer.daemon = True
        timer.start()
    # Start the game!
    pyglet.resource.path = ['../resources']
    pyglet.resource.reindex()
    game(scene=setup(test, ph_source, []))# if demo else [ae.finetune, lambda: ae.change_mode(ae.USING)]))
    cleanup()