from cocos.actions import *
from cocos.layer import *
from cocos.text import *

from ProgressBar import ProgressBar
from Resources import Colors, Fonts
from status import status

c = Colors()
f = Fonts()

SCORE_H = 96
PROG_H = 40
PROG_W = 200


class ScoreLayer(Layer):
    objectives = []
    def __init__(self):
        w, h = director.get_window_size()
        super(ScoreLayer, self).__init__()

        # transparent layer
        trblue = c.blue[0:3] + (100,) # <- transparency
        self.add(ColorLayer(*trblue, width=w, height=SCORE_H), z=-1)
        self.position = (0, h - SCORE_H)

        progress_bar = self.progress_bar = ProgressBar(width=PROG_W, height=PROG_H)
        dist = (SCORE_H-PROG_H)//2
        progress_bar.position = dist, dist
        self.add(progress_bar)

        labelSettings = dict(font_size=f.size_selected,
                             font_name=f.default,
                             color=c.white,
                             anchor_x='right',
                             anchor_y='center')
        self.score = Label('Score:', **labelSettings)
        self.score.position = (w-dist, SCORE_H//2)
        self.add(self.score)
        #lvl
        labelSettings['anchor_x']= 'center'
        self.lvl = Label('Lvl:', **labelSettings)
        self.lvl.position = (w//2, SCORE_H//2)
        self.add(self.lvl)

    def draw(self):
        super(ScoreLayer, self).draw()
        self.score.element.text = 'Score:%d' % status.score

        lvl = status.level_idx or 1
        self.lvl.element.text = 'Lvl: '+str(lvl)

class ObjectiveLayer(Layer):
    def __init__(self):
        w,h = director.window.get_size()
        super(ObjectiveLayer, self).__init__()
        # Transparent
        trblue = c.blue[0:3] + (50,) # <- transparency
        self.w = w-800
        self.h = h-SCORE_H
        self.add(ColorLayer(*trblue, width=self.w, height=self.h), z=-1)
        self.position = (0, 0)

        self.objectives_list = []
        self.objectives_labels = []

    def set_objectives(self, objectives):
        w, h = director.get_window_size()
        # Clear any previously set objectives
        for tile_type, sprite, count in self.objectives_list:
            self.remove(sprite)
        for count_label in self.objectives_labels:
            self.remove(count_label)
        self.objectives_list = objectives
        self.objectives_labels = []
        x = PROG_H+15
        y = self.h-PROG_H//2-32
        for tile_type, sprite, count in objectives:
            text_w = len(str(count))*8
            count_label_cfg = dict(font_size=20,
                                   font_name=f.default,
                                   color=c.white,
                                   bold=True,
                                   anchor_x='left',
                                   anchor_y='bottom')
            count_label = Label(str(count), **count_label_cfg)
            count_label.position = x-text_w, y-75
            self.add(count_label, z=2)
            self.objectives_labels.append(count_label)
            sprite.position = x, y
            sprite.scale = 0.85
            x += 75
            self.add(sprite)


class MessageLayer(Layer):
    def show_message(self, msg, callback=None, msg_duration=1):
        w, h = director.get_window_size()
        print

        self.msg = RichLabel(msg,
                             font_size=52,
                             font_name=f.default,
                             anchor_y='center',
                             anchor_x='center',
                             align='center',
                             multiline=True,
                             width=w)
        self.msg.position = (w // 2.0, h)

        self.add(self.msg)

        actions = Accelerate(MoveBy((0, -h / 2.0), duration=0.5))
        actions += \
            Delay(msg_duration - 1) + \
            Accelerate(MoveBy((0, -h / 2.0), duration=0.5)) + \
            Hide()

        if callback:
            actions += CallFunc(callback)

        self.msg.do(actions)


class HUD(Layer):
    def __init__(self):
        super(HUD, self).__init__()
        self.score_layer = ScoreLayer()
        self.objective_layer = ObjectiveLayer()
        self.add(self.score_layer)
        self.add(self.objective_layer)
        self.add(MessageLayer(), name='msg')

    def show_message(self, msg, callback=None, msg_duration=1):
        self.get('msg').show_message(msg, callback, msg_duration)

    def set_objectives(self, objectives):
        self.objective_layer.set_objectives(objectives)

    def update_time(self, time_percent):
        self.score_layer.progress_bar.set_progress(time_percent)