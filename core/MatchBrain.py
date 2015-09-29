__author__ = 'joren'

import logging
from datetime import datetime as dt

import pyglet.resource
from cocos import director
from cocos.scene import Scene
from cocos.layer import MultiplexLayer

from Menus import MainMenu
from LogHelpers import CustomHandler

version = 0.1

def setup():
    director.director.init(width=800, height=700, caption='MatchBrain')
    scene = Scene()
    scene.add(MultiplexLayer(
        MainMenu(),
    ), z = 1)
    return scene

def game(scene):
    director.director.run(scene)

def cleanup():
    pass

if __name__ == '__main__':
    # (Data) logging stuff
    guineaPig = raw_input('Enter nickname: ')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    rootHandler = logging.StreamHandler()
    rootLogger.addHandler(rootHandler)
    jsonLogger = logging.getLogger('json')
    jsonLogger.setLevel(logging.INFO)
    jsonHandler = CustomHandler(
        '../logs/'+guineaPig+dt.now().strftime('%Y-%m-%dT%H:%M')+'.json',
        {'nick': guineaPig, 'startTime': dt.now(), 'version': version})
    rootLogger.debug("Yo!", extra={"test": 3})
    jsonLogger.info({"test": "value", "other": {"more": 0}})
    # Start the game!
    pyglet.resource.path = ['../resources']
    pyglet.resource.reindex()
    game(scene=setup())
    cleanup()