__author__ = 'joren'

import logging
import threading
import time
from datetime import datetime as dt

import pyglet.resource
from cocos import director
from cocos.layer import MultiplexLayer
from cocos.scene import Scene

from LogHelpers import CustomHandler
from Menus import MainMenu
from signals.mindwave import BrainWaveSource
from signals.primitive import LogSink

version = 0.2

def setup(test=False):
    director.director.init(width=800, height=700, caption='MatchBrain')
    scene = Scene()
    scene.add(MultiplexLayer(
        MainMenu(test),
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
    if not demo:
        dataLogger = logging.getLogger('data')
        dataLogger.setLevel(logging.INFO)
        jsonHandler = CustomHandler(
            '../logs/'+nickname+dt.now().strftime('%Y-%m-%dT%H:%M')+'.json',
            {'nick': nickname, 'startTime': dt.now(), 'version': version})
        dataLogger.addHandler(jsonHandler)
        bwSource = BrainWaveSource()
        #bwSource.push()
        bwSink = LogSink(bwSource, logging.getLogger('data.brainwave'))
        def bwUpdate(source, sink, seconds):
            dataLogger.debug('before internal')
            def internal():
                #next_call = time.time()
                while True:
                    dataLogger.debug('internal 1')
                    #next_call = next_call + seconds
                    source.push()
                    dataLogger.debug('internal pushed')
                    sink.pull()
                    dataLogger.debug('internal pulled')
                    time.sleep(seconds) #abs(next_call - time.time())) #DIRTY
            return internal
        timer = threading.Thread(target = bwUpdate(bwSource, bwSink, 1))
        timer.daemon = True
        timer.start()
    # Start the game!
    pyglet.resource.path = ['../resources']
    pyglet.resource.reindex()
    game(scene=setup(test))
    cleanup()