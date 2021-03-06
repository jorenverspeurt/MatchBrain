import cocos
import cocos.layer
from cocos.director import director
from cocos.scene import Scene

from GameController import GameController
from GameModel import GameModel
from HUD import HUD
from Resources import Colors
from status import status

c = Colors()

__all__ = ['get_newgame']

class GameView(cocos.layer.ColorLayer):
    is_event_handler = True  #: enable director.window events

    def __init__(self, model, hud):
        super(GameView, self).__init__(*(c.blue[0:3]+(0,)))
        model.set_view(self)
        self.hud = hud
        self.model = model
        self.model.push_handlers(self.on_update_objectives
                                 , self.on_update_time
                                 , self.on_game_over
                                 , self.on_level_completed)
        self.model.start()
        self.hud.show_message('GET READY\n')

    def on_update_objectives(self):
        self.hud.set_objectives(self.model.objectives)

    def on_update_time(self, time_percent):
        self.hud.update_time(time_percent)

    def on_game_over(self):
        self.hud.show_message('GAME OVER\n', msg_duration=3, callback=lambda: director.pop())

    def on_level_completed(self):
        self.hud.show_message('LEVEL COMPLETED\n', msg_duration=3, callback=lambda: self.model.set_next_level())

    def on_message(self, msg):
        self.hud.show_message(msg, msg_duration=1)


def get_newgame(level = 1):
    status.score = 0
    status.level = level

    scene = Scene()
    model = GameModel(hud_offset=director.window.get_size()[0]-800) #MAGIC original width
    controller = GameController(model)
    # view
    hud = HUD()
    view = GameView(model, hud)

    # set controller in model
    model.set_controller(controller)

    # add controller
    scene.add(controller, z=1, name="controller")
    scene.add(hud, z=3, name="hud")
    scene.add(view, z=2, name="view")

    return scene