__author__ = 'joren'

class Colors(object):
    @property
    def red(self):
        return (204, 0, 68, 255)

    @property
    def orange(self):
        return (255, 170, 0, 255)

    @property
    def yellow(self):
        return (255, 255, 0, 255)

    @property
    def green(self):
        return (170, 255, 0, 255)

    @property
    def cyan(self):
        return (0, 255, 170, 255)

    @property
    def blue(self):
        return (0, 170, 255, 255)

    @property
    def purple(self):
        return (170, 0, 255, 255)

    @property
    def white(self):
        return (255, 255, 255, 255)

    @property
    def grey(self):
        return (62, 89, 102, 255)

    @property
    def black(self):
        return (0, 0, 0, 255)

class Fonts(object):
    sizes = {
        'title': 72,
        'item': 36,
        'selected_item': 42
    }

    @property
    def default(self):
        return "Droid Sans Mono"

    def size(self, type):
        return Fonts.sizes[type]

    @property
    def size_title(self):
        return Fonts.sizes['title']

    @property
    def size_item(self):
        return Fonts.sizes['item']

    @property
    def size_selected(self):
        return Fonts.sizes['selected_item']

