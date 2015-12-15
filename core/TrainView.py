
__author__ = 'joren'

import logging
from collections import namedtuple

import cocos
import cocos.layer
from cocos import director
from cocos.scene import Scene

from GameView import get_newgame
from HUD import MessageLayer

__all__ = ['get_newtrainer', 'phase_names']

Phase = namedtuple('Phase', ['name', 'msg', 'duration', 'callback'])
phase_names = ['DISTRACT', 'RELAXOPEN', 'RELAXCLOSED', 'CASUAL', 'INTENSE']

class TrainView(cocos.layer.ColorLayer):
    is_event_handler = True

    def __init__(self, scene, test=False, phase_source = None, train_end_callables = None):
        cocos.layer.ColorLayer.__init__(self, 0, 0, 0, 0)
        self.scene = scene
        self.msg = MessageLayer()
        longMsg = 60 if not test else 3
        shortMsg = 3 if not test else 1
        self.phases = [
            Phase('DISTRACT', "Look around,\nget distracted", longMsg, self.nextPhase),
            Phase('RELAXOPEN', "Relax (with eyes open)", longMsg, self.nextPhase),
            Phase('RELAXCLOSED', "Relax (with eyes closed)", longMsg, self.nextPhase),
            Phase('CASUAL', "Play casually\n(but towards the objective)", shortMsg, lambda: self.play(1)),
            Phase('INTENSE', "Match as fast as possible\n(forget the objective)", shortMsg, lambda: self.play(0))
        ]
        assert all(name in [phase.name for phase in self.phases] for name in phase_names) #selfcheck...
        if phase_source:
            phase_source.upstream = self.getPhase
        self.train_end_callables = train_end_callables or []
        self.phase = 0
        self.phaseLogger = logging.getLogger('data.train.phase')
        self.nextPhase()

    def getPhase(self):
        if self.phase < len(phase_names):
            return phase_names[self.phase]
        else:
            return "DISTRACT"

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
            for c in self.train_end_callables:
                c()
            director.director.pop()

    def play(self, level):
        game = get_newgame(level)
        def on_exit():
            self.nextPhase()
        game.on_exit = on_exit
        director.director.push(game)


def get_new_trainer(test=False, phase_source=None, train_end_callables=[]):
    scene = Scene()
    view = TrainView(scene, test=test, phase_source=phase_source, train_end_callables=train_end_callables)
    scene.add(view.msg, z=2, name='hud')
    scene.add(view, z=1, name='trainview')
    return scene
