import pyglet
from cocos.director import director
from cocos.menu import *
from cocos.scenes.transitions import *

from Resources import Colors
from Resources import Fonts

c = Colors()
f = Fonts()

class MainMenu(Menu):
    def __init__(self, test=False):
        super(MainMenu, self).__init__('MatchBrain')
        self.test = test

        defont = f.default

        self.font_title['font_name'] = defont
        self.font_title['font_size'] = f.size_title
        self.font_title['color'] = c.white

        self.font_item['font_name'] = defont
        self.font_item['font_size'] = f.size_item
        self.font_item['color'] = c.purple

        self.font_item_selected['font_name'] = defont
        self.font_item_selected['font_size'] = f.size_selected
        self.font_item_selected['color'] = c.green

        # example: menus can be vertical aligned and horizontal aligned
        self.menu_anchor_y = CENTER
        self.menu_anchor_x = CENTER

        items = []

        items.append(MenuItem('New Game', self.on_new_game))
        #items.append( MenuItem('Options', self.on_options) )
        #items.append( MenuItem('Scores', self.on_scores) )
        items.append(MenuItem('Train', self.on_train))
        items.append(MenuItem('Quit', self.on_quit))

        self.create_menu(items, shake(), shake_back())

    def on_new_game(self):
        import GameView
        import logging

        phaseLogger = logging.getLogger('data.train.phase')
        phaseLogger.info('NEWGAME')
        director.push(FlipAngular3DTransition(
            GameView.get_newgame(), 1.5))

    def on_options(self):
        self.parent.switch_to(1)

    def on_train(self):
        import TrainView

        director.push(TrainView.get_new_trainer(self.test))

    def on_scores(self):
        self.parent.switch_to(2)

    def on_quit(self):
        pyglet.app.exit()
