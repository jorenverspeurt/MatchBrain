import logging

from cocos.layer import Layer
import pyglet.window.key


class GameController(Layer):

    is_event_handler = True #: enable pyglet's events

    def __init__(self, model):
        super(GameController, self).__init__()
        self.model = model
        self.keyLogger = logging.getLogger('data.events.key')
        self.mouseLogger = logging.getLogger('data.events.mouse')

    def on_mouse_press(self, x, y, buttons, modifiers):
        self.mouseLogger.info({'press': {'x': x, 'y': y, 'buttons': buttons}})
        self.model.on_mouse_press(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouseLogger.info({'drag': {'x': x, 'y': y, 'dx': dx, 'dy': dy, 'buttons': buttons}})
        self.model.on_mouse_drag(x, y)

    def on_key_press(self, key, modifiers):
        symbol = pyglet.window.key.symbol_string(key)
        self.keyLogger.info(symbol)
