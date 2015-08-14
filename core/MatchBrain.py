__author__ = 'joren'
from os.path import exists, join

import pyglet.resource
from cocos import director
from cocos.scene import Scene
from cocos.layer import MultiplexLayer
from Menus import MainMenu

def setup():
    director.director.init(width=800, height=650, caption='MatchBrain')
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
    pyglet.resource.path = ['../resources']
    pyglet.resource.reindex()
    testIm = pyglet.resource.image('46.png')
    game(scene=setup())
    cleanup()