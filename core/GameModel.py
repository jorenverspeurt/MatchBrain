__all__ = ['GameModel']

import logging
from glob import glob
from math import log
from random import choice, randint

import pyglet
from cocos.actions import *
from cocos.sprite import Sprite

from status import status

CELL_WIDTH, CELL_HEIGHT = 100, 100
ROWS_COUNT, COLS_COUNT = 6, 8

# Game State Values
WAITING_PLAYER_MOVEMENT = 'waiting_player_movement'
PLAYER_DOING_MOVEMENT = 'player_doing_movement'
SWAPPING_TILES = 'swapping_tiles'
IMPLODING_TILES = 'imploding_tiles'
DROPPING_TILES = 'dropping_tiles'
NEXT_LEVEL = 'next_level'
GAME_OVER = 'game_over'

class GameEvent(object):
    def __init__(self
                 ,type = ''
                 ,effect = lambda x: x):
        self.type = type
        self.effect = effect

class GameCondition(object):
    def __init__(self
                 ,type = ''
                 ,value = None
                 ,condition = lambda x: True):
        self.value = value
        self.condition = condition
        self.type = type
        self.satisfied = False

    def handle(self, event):
        if not self.satisfied and event.type == self.type:
            self.value = event.effect(self.value)
            self.satisfied = self.condition(self.value)
        return self

class MultipleConditions(GameCondition):
    def __init__(self, conds):
        GameCondition.__init__(self, 'any', conds, lambda cs: all(c.satisfied for c in cs))

    def handle(self, event):
        if not self.satisfied:
            self.value = map(lambda v: v.handle(event), self.value)
            self.satisfied = self.condition(self.value)
        return self

class GameAction(object):
    def __init__(self
                 ,with_board = None
                 ,with_drops = None
                 ,with_score = None
                 ,message = ""):
        self.with_board = with_board or (lambda x: x)
        self.with_drops = with_drops or (lambda x: x)
        self.with_score = with_score or (lambda x: x)
        self.message = message

class Objective(object):
    def __init__(self,
                 parent,
                 aop = 'action',
                 time = 30,
                 start = None,
                 stop = GameCondition(),
                 cleanup = GameAction()):
        self.parent = parent
        self.aop = aop
        self.time = time
        self.stop = stop
        self.cleanup = cleanup
        if start:
            for ga in start:
                parent.register_game_action(ga)
        parent.add_time(time)

    def process_events(self, events):
        for event in events:
            self.stop.handle(event)
        if self.stop.satisfied:
            self.parent.register_game_action(self.cleanup)
            return True
        else:
            return False

class GameModel(pyglet.event.EventDispatcher):
    def __init__(self, hud_offset = 0):
        super(GameModel, self).__init__()
        self.HUD_OFFSET = hud_offset
        self.tile_grid = {}  # Dict emulated sparse matrix, key: tuple(x,y), value : tile_type
        self.imploding_tiles = []  # List of tile sprites being imploded, used during IMPLODING_TILES
        self.dropping_tiles = []  # List of tile sprites being dropped, used during DROPPING_TILES
        self.swap_start_pos = None  # Position of the first tile clicked for swapping
        self.swap_end_pos = None  # Position of the second tile clicked for swapping
        self.available_tiles = [s.replace('../resources/', '') for s in glob('../resources/*.png')]
        self._game_state = WAITING_PLAYER_MOVEMENT
        self.objectives = []
        self.stateLogger = logging.getLogger('data.game.state')
        self.scoreLogger = logging.getLogger('data.game.score')
        self.objectiveLogger = logging.getLogger('data.game.objective')
        self.max_play_time = 60
        self.ticking = False
        self.board_actions = []
        self.drop_actions = []
        self.score_actions = []
        self.message_actions = []

    def set_tick(self, on_off, interval=1):
        """
        To prevent multiple tick timers from being scheduled all access to them must go through this method
        :param on_off: Whether the model should have a scheduled tick timer or not
        :param interval: Optional interval length
        """
        if self.ticking != on_off:
            if on_off:
                pyglet.clock.schedule_interval(self.time_tick, interval)
            else:
                pyglet.clock.unschedule(self.time_tick)
            self.ticking = on_off

    @property
    def game_state(self):
        """ The current state of the game model's state machine """
        return self._game_state

    @game_state.setter
    def game_state(self, value):
        self._game_state = value
        self.stateLogger.info(value)

    @game_state.deleter
    def game_state(self):
        del self._game_state

    def start(self):
        self.set_next_level()

    def set_next_level(self):
        self.play_time = self.max_play_time
        for elem in self.imploding_tiles + self.dropping_tiles:
            if elem in self.view.children_names.values():
                self.view.remove(elem)
        self.fill_with_random_tiles()
        self.set_objectives()
        self.set_tick(False)
        self.set_tick(True)
        self.game_state = WAITING_PLAYER_MOVEMENT
        if status.level == 0:
            status.level_idx = "INTENSE"
        else:
            status.level_idx = status.level

    def time_tick(self, delta):
        self.play_time -= 1
        self.dispatch_event("on_update_time", self.play_time/float(self.max_play_time))
        if self.message_actions:
            msg_act = self.message_actions.pop(0)
            msg = msg_act.message
            if msg:
                self.dispatch_event("on_message", msg)
        if self.play_time == 0:
            self.set_tick(False)
            self.game_state = GAME_OVER
            self.dispatch_event("on_game_over")

    def set_objectives(self):
        if True:
            objectives = []
            while len(objectives) < 3:
                tile_type = choice(self.available_tiles)
                sprite = self.tile_sprite(tile_type, (0, 0))
                max = 100 if status.level == 0 else round(5+10*log(status.level))
                count = randint(1, max)
                if tile_type not in [x[0] for x in objectives]:
                    objectives.append([tile_type, sprite, count])
            self.objectiveLogger.info({ 'type': "3kinds"
                                      , 'duration': 60
                                      , '3kinds': {obj[0]: obj[2] for obj in objectives}})
            self.objectives = objectives
            self.dispatch_event("on_update_objectives")
        else:
            types = self.get_objective_types()

            objectives = map(lambda t: self.objective_for_difficulty(t, self.difficulty_for(t)))


    def get_objective_types(self):
        # TODO tie this into the learner system
        return ['3kinds', 'puzzle0']

    def objective_for_difficulty(self, type, difficulty):
        if type == '3kinds':
            value = {tile: (count, sprite) for _ in xrange(3) for tile in (choice(self.available_tiles))}

    def fill_with_random_tiles(self):
        """
        Fills the tile_grid with random tiles
        """
        #j BUG: I think sometimes the game tries to remove tiles that have already been removed
        #       due to a match... This doesn't occur very often. Probably some race condition.
        # Possible integral fix: make some sort of set-queue thing for removal jobs, only adding them once
        # Or make remove fully idempotent, somehow
        for elem in [x[1] for x in self.tile_grid.values() if x]:
            try:
                if elem:
                    self.view.remove(elem)
            except Exception as e:
                self.stateLogger.exception(e.message)
        tile_grid = {}
        # Fill the data matrix with random tile types
        # TODO make this depend on the difficulty / objective / ... ?
        while True:  # Loop until we have a valid table (no imploding lines)
            for x in range(COLS_COUNT):
                for y in range(ROWS_COUNT):
                    tile_type, sprite = choice(self.available_tiles), None
                    tile_grid[x, y] = tile_type, sprite
            if len(self.get_same_type_lines(tile_grid)) == 0:
                break
            tile_grid = {}

        # Build the sprites based on the assigned tile type
        for key, value in tile_grid.iteritems():
            tile_type, sprite = value
            sprite = self.tile_sprite(tile_type, self.to_display(key))
            tile_grid[key] = tile_type, sprite
            self.view.add(sprite)

        self.tile_grid = tile_grid

    def swap_elements(self, elem1_pos, elem2_pos):
        tile_type, sprite = self.tile_grid[elem1_pos]
        self.tile_grid[elem1_pos] = self.tile_grid[elem2_pos]
        self.tile_grid[elem2_pos] = tile_type, sprite

    def implode_lines(self):
        """
        :return: Implodes lines with more than 3 elements of the same type
        """
        implode_count = {}
        for x, y in self.get_same_type_lines(self.tile_grid):
            tile_type, sprite = self.tile_grid[x, y]
            self.tile_grid[x, y] = None
            self.imploding_tiles.append(sprite)  # Track tiles being imploded
            sprite.do(ScaleTo(0, 0.5) | RotateTo(180, 0.5) + CallFuncS(self.on_tile_remove))  # Implode animation
            implode_count[tile_type] = implode_count.get(tile_type, 0) + 1
        # Decrease counter for tiles matching objectives
        for elem in self.objectives:
            if elem[0] in implode_count:
                Scale = ScaleBy(1.5, 0.2)
                elem[2] = max(0, elem[2]-implode_count[elem[0]])
                elem[1].do((Scale + Reverse(Scale))*3)
        # Remove objectives already completed
        self.objectives = [elem for elem in self.objectives if elem[2] > 0]
        self.objectiveLogger.info({ 'type': "3kinds"
                                  , 'duration': self.play_time
                                  , '3kinds': {obj[0]: obj[2] for obj in self.objectives}})
        if len(self.imploding_tiles) > 0:
            self.game_state = IMPLODING_TILES  # Wait for the implosion animation to finish
            self.set_tick(False)
        else:
            if not self.game_state == NEXT_LEVEL:
                self.game_state = WAITING_PLAYER_MOVEMENT
            self.set_tick(True)
        return self.imploding_tiles

    def drop_groundless_tiles(self):
        """
        Walk on all columns, from bottom to up:
            a) count gap or move down pieces for gaps already counted
            b) on top drop as much tiles as gaps counted
        :return:
        """
        tile_grid = self.tile_grid

        for x in range(COLS_COUNT):
            gap_count = 0
            for y in range(ROWS_COUNT):
                if tile_grid[x, y] is None:
                    gap_count += 1
                elif gap_count > 0:  # Move from y to y-gap_count
                    tile_type, sprite = tile_grid[x, y]
                    if gap_count > 0:
                        sprite.do(MoveTo(self.to_display((x, y - gap_count)), 0.3 * gap_count))
                    tile_grid[x, y - gap_count] = tile_type, sprite
            for n in range(gap_count):  # Drop as much tiles as gaps counted
                tile_type = choice(self.available_tiles)
                sprite = self.tile_sprite(tile_type, self.to_display((x, y + n + 1)))
                tile_grid[x, y - gap_count + n + 1] = tile_type, sprite
                sprite.do(
                    MoveTo(self.to_display((x, y - gap_count + n + 1)), 0.3 * gap_count) + CallFuncS(
                        self.on_drop_completed))
                self.view.add(sprite)
                self.dropping_tiles.append(sprite)

    def on_drop_completed(self, sprite):
        self.dropping_tiles.remove(sprite)
        if len(self.dropping_tiles) == 0:  # All tile dropped
            self.implode_lines()  # Check for new implosions

    def on_tile_remove(self, sprite):
        status.score += len(self.imploding_tiles)//3
        self.scoreLogger.info(status.score)
        self.imploding_tiles.remove(sprite)
        self.view.remove(sprite)
        if len(self.imploding_tiles) == 0:  # Implosion complete, drop tiles to fill gaps
            self.dispatch_event("on_update_objectives")
            self.drop_groundless_tiles()
            if len(self.objectives) == 0:
                self.set_tick(False)
                if status.level:
                    status.level += 1
                self.game_state = NEXT_LEVEL
                self.dispatch_event("on_level_completed")

    def set_controller( self, controller):
        self.controller = controller

    def set_view( self, view):
        self.view = view

    def tile_sprite(self, tile_type, pos):
        """
        :param tile_type: numeric id, must be in the range of available images
        :param pos: sprite position
        :return: sprite built form tile_type
        """
        sprite = Sprite(tile_type)
        sprite.position = pos
        sprite.scale = 1
        sprite.name = tile_type #MONKEY!
        return sprite

    def on_tiles_swap_completed(self):
        self.game_state = DROPPING_TILES
        if len(self.implode_lines()) == 0:
            # No lines imploded, roll back the game play

            # Start swap animation for both objects
            tile_type, sprite = self.tile_grid[self.swap_start_pos]
            sprite.do(MoveTo(self.to_display(self.swap_end_pos), 0.4))
            tile_type, sprite = self.tile_grid[self.swap_end_pos]
            sprite.do(MoveTo(self.to_display(self.swap_start_pos), 0.4) + CallFunc(self.on_tiles_swap_back_completed))

            # Revert on the grid
            self.swap_elements(self.swap_start_pos, self.swap_end_pos)
            self.game_state = SWAPPING_TILES

    def on_tiles_swap_back_completed(self):
        if not self.game_state == NEXT_LEVEL:
            self.game_state = WAITING_PLAYER_MOVEMENT

    def to_display(self, (row, col)):
        """
        :param row:
        :param col:
        :return: (x, y) from display corresponding coordinates from the bi-dimensional ( row, col) array position
        """
        return CELL_WIDTH / 2 + row * CELL_WIDTH + self.HUD_OFFSET, CELL_HEIGHT / 2 + col * CELL_HEIGHT

    def to_model_pos(self, (view_x, view_y)):
        return (view_x - self.HUD_OFFSET) / CELL_WIDTH, view_y / CELL_HEIGHT

    def get_same_type_lines(self, tile_grid, min_count=3):
        """
        Identify vertical and horizontal lines composed of min_count consecutive elements
        :param min_count: minimum consecutive elements to identify a line
        """
        all_line_members = []

        # Check for vertical lines
        for x in range(COLS_COUNT):
            same_type_list = []
            last_tile_type = None
            for y in range(ROWS_COUNT):
                tile_type, sprite = tile_grid[x, y]
                if last_tile_type == tile_type:
                    same_type_list.append((x, y))
                if tile_type != last_tile_type or y == ROWS_COUNT - 1:  # Line end because type changed or edge reached
                    if len(same_type_list) >= min_count:
                        all_line_members.extend(same_type_list)
                    last_tile_type = tile_type
                    same_type_list = [(x, y)]

        # Check for horizontal lines
        for y in range(ROWS_COUNT):
            same_type_list = []
            last_tile_type = None
            for x in range(COLS_COUNT):
                tile_type, sprite = tile_grid[x, y]
                if last_tile_type == tile_type:
                    same_type_list.append((x, y))
                if tile_type != last_tile_type or x == COLS_COUNT - 1:  # Line end because of type change or edge reached
                    if len(same_type_list) >= min_count:
                        all_line_members.extend(same_type_list)
                    last_tile_type = tile_type
                    same_type_list = [(x, y)]

        # Remove duplicates
        all_line_members = list(set(all_line_members))
        return all_line_members

    def on_key_press(self, key):
        # To be able te recover if a test subject triggers a bug where the playing field freezes
        if key == pyglet.window.key.NUM_ENTER:
            self.game_state = WAITING_PLAYER_MOVEMENT

    def on_mouse_press(self, x, y):
        if self.game_state == WAITING_PLAYER_MOVEMENT or self.game_state == PLAYER_DOING_MOVEMENT:
            self.swap_start_pos = self.to_model_pos((x, y))
            self.game_state = PLAYER_DOING_MOVEMENT

    def on_mouse_drag(self, x, y):
        if self.game_state != PLAYER_DOING_MOVEMENT:
            return

        start_x, start_y = self.swap_start_pos
        self.swap_end_pos = new_x, new_y = self.to_model_pos((x, y))

        distance = abs(new_x - start_x) + abs(new_y - start_y)  # horizontal + vertical grid steps

        # Ignore movement if not at 1 step away from the initial position
        if new_x < 0 or new_y < 0 or distance != 1:
            return

        # Start swap animation for both objects
        tile_type, sprite = self.tile_grid[self.swap_start_pos]
        sprite.do(MoveTo(self.to_display(self.swap_end_pos), 0.4))
        tile_type, sprite = self.tile_grid[self.swap_end_pos]
        sprite.do(MoveTo(self.to_display(self.swap_start_pos), 0.4) + CallFunc(self.on_tiles_swap_completed))

        # Swap elements at the board data grid
        self.swap_elements(self.swap_start_pos, self.swap_end_pos)
        self.game_state = SWAPPING_TILES

    def dump_table(self):
        """
        :return: Prints the play table, for debug
        """
        for y in range(ROWS_COUNT - 1, -1, -1):
            line_str = ''
            for x in range(COLS_COUNT):
                line_str += str(self.tile_grid[x, y][0])
            print line_str

GameModel.register_event_type('on_update_objectives')
GameModel.register_event_type('on_update_time')
GameModel.register_event_type('on_game_over')
GameModel.register_event_type('on_level_completed')
GameModel.register_event_type('on_message')
