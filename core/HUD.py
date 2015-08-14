from cocos.layer import *

from cocos.text import *
from cocos.actions import *
from ProgressBar import ProgressBar
from status import status

from Resources import Colors, Fonts
c = Colors()
f = Fonts()

SCORE_H = 96
PROG_H = 40
PROG_W = 200

class BackgroundLayer(Layer):
    def __init__(self):
        super(BackgroundLayer, self).__init__()
        # self.img = pyglet.resource.image('background.png')

    def draw(self):
        pass
        # glPushMatrix()
        # self.transform()
        #self.img.blit(0,0)
        #glPopMatrix()


class ScoreLayer(Layer):
    objectives = []
    def __init__(self):
        w, h = director.get_window_size()
        super(ScoreLayer, self).__init__()

        # transparent layer
        trblue = c.blue[0:3] + (100,)
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
        self.score.position = (w-dist, SCORE_H/2)
        self.add(self.score)
        self.lvl = Label('Lvl:', **labelSettings)
        self.lvl.position = (450, 0)
        #self.add(self.lvl)

        self.objectives_list = []
        self.objectives_labels = []

    def set_objectives(self, objectives):
        w, h = director.get_window_size()
        # Clear any previously set objectives
        for tile_type, sprite, count in self.objectives:
            self.remove(sprite)
        for count_label in self.objectives_labels:
            self.remove(count_label)
        self.objectives = objectives
        self.objectives_labels = []
        x = w/2-150/2
        for tile_type, sprite, count in objectives:
            text_w = len(str(count))*7
            count_label_set = dict(font_size=14,
                                   font_name=f.default,
                                   color=c.white,
                                   bold=True,
                                   anchor_x='left',
                                   anchor_y='bottom')
            count_label = Label(str(count), **count_label_set)
            count_label.position = x-text_w, 7
            self.add(count_label, z=2)
            self.objectives_labels.append(count_label)
            count_label_set['font_size'] = 16
            count_label = Label(str(count), **count_label_set)
            count_label.position = x-text_w-1, 8
            self.add(count_label, z=1)
            self.objectives_labels.append(count_label)
            sprite.position = x, 24
            sprite.scale = 0.5
            x += 50
            self.add(sprite)

    def draw(self):
        super(ScoreLayer, self).draw()
        self.score.element.text = 'Score:%d' % status.score

        lvl = status.level_idx or 0
        self.lvl.element.text = 'Lvl:%d' % lvl


class MessageLayer(Layer):
    def show_message(self, msg, callback=None, msg_duration=1):
        w, h = director.get_window_size()

        self.msg = Label(msg,
                         font_size=52,
                         font_name=f.default,
                         anchor_y='center',
                         anchor_x='center')
        self.msg.position = (w // 2.0, h)

        self.add(self.msg)

        actions = Accelerate(MoveBy((0, -h / 2.0), duration=msg_duration/2))
        actions += \
            Delay(1) + \
            Accelerate(MoveBy((0, -h / 2.0), duration=msg_duration/2)) + \
            Hide()

        if callback:
            actions += CallFunc(callback)

        self.msg.do(actions)


class HUD(Layer):
    def __init__(self):
        super(HUD, self).__init__()
        self.score_layer = ScoreLayer()
        self.add(self.score_layer)
        self.add(MessageLayer(), name='msg')

    def show_message(self, msg, callback=None, msg_duration=1):
        self.get('msg').show_message(msg, callback, msg_duration)

    def set_objectives(self, objectives):
        self.score_layer.set_objectives(objectives)

    def update_time(self, time_percent):
        self.score_layer.progress_bar.set_progress(time_percent)