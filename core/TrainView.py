__author__ = 'joren'

import cocos
import cocos.layer

__all__ = ['get_newtrainer']

class TrainView(cocos.layer.ColorLayer):
    is_event_handler = True

    def __init__(self):
        cocos.layer.MultiplexLayer.__init__(self, (0,0,0,0))
        
