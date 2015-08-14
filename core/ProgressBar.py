import cocos
from pyglet.gl import *
from Resources import Colors
c = Colors()

class ProgressBar(cocos.cocosnode.CocosNode):

    def __init__(self, width, height):
        super(ProgressBar, self).__init__()
        self.width, self.height = width, height
        self.vertexes_in = [(0, 0, 0), (width, 0, 0), (width, height, 0), (0, height, 0)]
        self.vertexes_out = [(-2, -2, 0), (width+2, -2, 0), (width+2, height+2, 0), (-2, height+2, 0)]

    def set_progress(self, percent):
        width = int(self.width * percent)
        height = self.height
        self.vertexes_in = [(0, 0, 0), (width, 0, 0), (width, height, 0), (0, height, 0)]

    def draw(self):
        glPushMatrix()
        self.transform()
        glBegin(GL_QUADS)
        glColor4ub(*c.white)
        for v in self.vertexes_out:
            glVertex3i(*v)
        glColor4ub(*c.green)
        for v in self.vertexes_in:
            glVertex3i(*v)
        glEnd()
        glPopMatrix()