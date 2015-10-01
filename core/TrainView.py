
__author__ = 'joren'

import logging
from collections import namedtuple

import cocos
import cocos.layer
from cocos.scene import Scene
from cocos import director

from HUD import MessageLayer
from GameView import get_newgame

__all__ = ['get_newtrainer']

Phase = namedtuple('Phase', ['name', 'msg', 'duration', 'callback'])

class TrainView(cocos.layer.ColorLayer):
    is_event_handler = True

    def __init__(self, scene):
        cocos.layer.ColorLayer.__init__(self, 0, 0, 0, 0)
        self.scene = scene
        self.msg = MessageLayer()
        self.phases = [
            Phase('DISTRACT', "Look around,\nget distracted", 1, self.nextPhase),
            Phase('RELAXOPEN', "Relax (with eyes open)", 1, self.nextPhase),
            Phase('RELAXCLOSED', "Relax (with eyes closed)", 1, self.nextPhase),
            Phase('CASUAL', "Play casually\n(but towards the objective)", 1, lambda: self.play(1)),
            Phase('INTENSE', "Match as fast as possible\n(forget the objective)", 1, lambda: self.play(0))
        ]
        self.phase = 0
        self.phaseLogger = logging.getLogger('data.train.phase')
        self.nextPhase()

    def nextPhase(self):
        if not self.phase == len(self.phases):
            current = self.phases[self.phase]
            self.phase += 1
            self.phaseLogger.info(current.name)
            self.msg.show_message(
                current.msg,
                callback=current.callback,
                msg_duration=current.duration)
        else:
            director.director.pop()

    def play(self, level):
        game = get_newgame(level)
        def on_exit():
            self.nextPhase()
        game.on_exit = on_exit
        director.director.push(game)


def get_new_trainer():
    scene = Scene()
    view = TrainView(scene)
    scene.add(view.msg, z=2, name='hud')
    scene.add(view, z=1, name='trainview')
    return scene


