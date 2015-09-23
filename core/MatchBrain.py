__author__ = 'joren'
from os.path import exists, join

import pyglet.resource
from cocos import director
from cocos.scene import Scene
from cocos.layer import MultiplexLayer
from Menus import MainMenu

import logging
from datetime import datetime

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
    rootLogger = logging.getLogger('')
    rootLogger.setLevel(logging.DEBUG)
    rootHandler = logging.StreamHandler()
    rootFormatter =
    jsonLogger = logging.getLogger('json')
    jsonLogger.setLevel(logging.INFO)
    jsonRootFormatter =
    pyglet.resource.path = ['../resources']
    pyglet.resource.reindex()
    game(scene=setup())
    cleanup()