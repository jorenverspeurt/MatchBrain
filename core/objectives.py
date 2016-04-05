from glob import glob
from random import choice, randint, sample

objective_types = ['3kinds', 'score', 'drop', 'number']
def_time = 45
# Bad bad bad. TODO find a way to keep this the same as the one in GameModel (separate python file?)
normal_tiles = [s.replace('../resources/', '') for s in glob('../resources/n-*.png')]
special_tiles = [s.replace('../resources/', '') for s in glob('../resources/s-*.png')]
other_tiles = [s.replace('../resources/', '') for s in glob('../resources/o-*.png')]

def insert_other(drops):
    index = randint(0, len(drops)-1)
    return [drop if i!=index else choice(other_tiles) for (i,drop) in enumerate(drops)]

def remove_others(grid):
    return [row
            if i!=len(grid)-1
            else [choice(normal_tiles)
                  if col_e in other_tiles
                  else col_e
                  for col_e in grid[-1]]
            for (i,row) in enumerate(grid)]

def insert_special(grid):
    row_i = randint(0, len(grid)-1)
    col_i = randint(0, len(grid[row_i]))
    return [row if i!=row_i else [col_e if j!=col_i else choice(special_tiles)
                                  for (j,col_e) in enumerate(grid[row_i])]
            for (i,row) in enumerate(grid)]

class ObjectiveType(object):
    all_types = {}
    by_cat = {}
    all_keys = {"score", "special_row", "combo"}
    all_keys.update(set(normal_tiles))

    def __init__(self, name, category, pre_actions = None, post_actions = None, fixed_time = None, fixed_value = None):
        self.name = name
        self.all_types[name] = self
        self.category = category
        self.by_cat[category] = name
        self.pre_actions = pre_actions
        self.post_actions = post_actions
        self.fixed_time = fixed_time
        self.fixed_value = fixed_value

    def instantiate(self, parent, difficulty):
        return Objective(parent,
                         self.category,
                         self.get_time(difficulty),
                         self.pre_actions,
                         OCondition(self.get_value(difficulty)),
                         self.post_actions)

    def get_time(self, difficulty):
        return self.fixed_time

    def get_value(self, difficulty):
        result = self.get_defaults()
        result.update(self._get_value(difficulty))
        return result

    def get_defaults(self):
        return {key: 0 for key in self.all_keys}

    def _get_value(self, difficulty):
        return self.fixed_value

    def display(self):
        return None

class ThreeKindsO(ObjectiveType):
    def __init__(self):
        ObjectiveType.__init__(self, '3kinds', 'action', None, None, 45, None)

    def _get_value(self, difficulty):
        min = difficulty*4
        max = difficulty*8
        total = difficulty*3*6
        counts = []
        counts.append(randint(min, max))
        counts.append(randint(min, max))
        counts.append(total-counts[0]-counts[1])
        value = {tile: (count, GameModel.tile_sprite(tile, (0,0)))
                 for (tile, count) in zip(sample(normal_tiles, 3), counts)}
        return value

class ScoreO(ObjectiveType):
    def __init__(self):
        ObjectiveType.__init__(self, 'score', 'action', None, None, 45, None)

    def _get_value(self, difficulty):
        return {"score": 20*difficulty}

class DropO(ObjectiveType):
    def __init__(self):
        ObjectiveType.__init__(self, 'drop', 'puzzle',
                               GameAction(with_drops=insert_other),
                               GameAction(with_board=remove_others),
                               None, 7)

class OEvent(object):
    def __init__(self, affects, function = lambda x: x):
        self.affects = affects
        self.function = function

    def change(self, value):
        value[self.affects] = self.function(value[self.affects])
        return value

class OCondition(object):
    def __init__(self
                ,value = None
                ,condition = lambda v: all(e<=0 for e in v)):
        self.value = value
        self.condition = condition
        self.satisfied = False

    def handle(self, event):
        if not self.satisfied:
            self.value = event.change(self.value)
            self.satisfied = self.condition(self.value)
        return self

class OMultipleConditions(OCondition):
    def __init__(self, conds):
        OCondition.__init__(self, conds, lambda cs: all(c.satisfied for c in cs))

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
    def __init__(self
                ,parent
                ,aop = 'action'
                ,time = 30
                ,start_actions = None
                ,stop = OCondition()
                ,cleanup_actions = None):
        self.parent = parent
        self.aop = aop
        self.time = time
        self.stop = stop
        self.cleanups = cleanup_actions
        if start_actions:
            for ga in start_actions:
                parent.register_game_action(ga)
        parent.add_time(time)

    def process_events(self, events):
        for event in events:
            self.stop.handle(event)
        if self.stop.satisfied:
            for cleanup in self.cleanups:
                self.parent.register_game_action(cleanup)
            return True
        else:
            return False

class ObjectiveMaker(object):
    #Modes
    STATIC = 'static'
    INTERNAL = 'internal'
    FEEDBACK = 'feedback'
    FULL = 'full'

    by_cat = {'action': ['3kinds', 'score']
             ,'puzzle': ['drop', 'number']}
    by_type = {'3kinds': 'action', 'score': 'action', 'drop': 'puzzle', 'number': 'puzzle'}
    mk = {'3kinds': lambda p, t, v: Objective(p, 'action', t,
                                              None, OCondition(value=v), None)
         ,'score': lambda p, t, v: Objective(p, 'action', t,
                                             None, OCondition(value=v), None)
         ,'drop': lambda p, t, v: Objective(p, 'puzzle', t,
                                            GameAction(with_drops=insert_other),
                                            OCondition(value=v),
                                            GameAction(with_board=remove_others))
         ,'number': lambda p, t, v: Objective(p, 'puzzle', t,
                                              None, OCondition(value=v), None)}
    # Yeah yeah I know I should have made an ObjectiveType class and done all of this that way...
    assert set(objective_types) == set(mk.keys())
    assert set(objective_types) == set(by_type.keys())
    assert set(objective_types) == set(sum(by_cat.values(), []))

    def __init__(self
                ,parent
                ,player_source = None
                ,game_source = None):
        if not (player_source or game_source):
            self.mode = self.STATIC
        elif game_source and not player_source:
            self.mode = self.INTERNAL
        elif player_source and not game_source:
            self.mode = self.FEEDBACK
        else:
            self.mode = self.FULL
        self.parent = parent
        self.player_source = player_source
        self.game_source = game_source

    def get_objectives_for_level(self, level):
        if self.mode == self.STATIC or self.mode == self.INTERNAL:
            cats = ['action', 'puzzle']
        else:
            cats = self.player_specific_cats()
        return map(lambda cat: self.obj_for_diff(choice(self.by_cat[cat]), self.difficulty_for(cat, level)), cats)

    def player_specific_cats(self):
        # TODO implement
        raise NotImplementedError()

    def obj_for_diff(self, category, difficulty):
        # TODO measure whether these make sense
        def score_diff(diff):
            return 20*diff
        def drop_diff(diff):
            return 60-5*diff
        def num_time_diff(diff):
            return (difficulty//3+3, 40-(difficulty%3)*10)
        # TODO load the above functions from somewhere else
        if type == '3kinds':
            # variable count, fixed time
            value = {tile: (count, tile_sprite(tile, (0,0)))
                     for (tile, count) in zip(sample(normal_tiles, 3), kinds_counts_diff(difficulty)) }
        elif type == 'score_fixtime':
            # variable score, fixed time
            value = {"score": score_diff(difficulty), "aop": 'action'}
        elif type == 'drop':
            # drop from top to bottom, variable time
            value = {"special_row": 6, "time": drop_diff(difficulty), "aop": 'puzzle'}
        elif type == 'make_number_time':
            # make a combo with a specific size, in a specified time
            ntd = num_time_diff(difficulty)
            value = {"combo": ntd[0], "time": ntd[1], "aop": 'puzzle'}
        else:
            raise ValueError("Unknown objective type")
        result = {"time": 45, "special_row": 0, "combo": 0, "score": 0}
        result.update({tile: 0 for tile in normal_tiles})
        result.update(value)
        return Objective(self.parent, value['aop'], value['time'], )


