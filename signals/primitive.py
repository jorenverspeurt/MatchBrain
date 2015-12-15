from __future__ import print_function

import logging
import random
import string
import sys
import time
from threading import Thread

__author__ = 'joren'

"""
Primitives for signal handling (kinda sorta FRP style)
Groups of signals are threaded through SignalBlocks. SignalBlocks consist of a collection of Sources, (Transformers) and Sinks.
Sources pull values from a provided method and Sinks push them out.
Transformers take a list of sources (Sources or Transformers) as inputs and a transformation to produce 1 output.
"""

debuggables = logging.getLogger('debuggables')
debuggables.setLevel(logging.DEBUG)
debuggables.addHandler(logging.StreamHandler(sys.stdout))

def randomName(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

class Debuggable(object):
    """
    Things that have a name and a signal or value that is logged based on the debug parameter.
    """

    def __init__(self, name, debug):
        self._name = name
        self._debug = debug
        self.logger = logging.getLogger('debuggables.'+name)
        self.logger.setLevel(logging.DEBUG)

    def getName(self):
        return self._name

    def debug(self, value = None):
        self.logger.debug(value)


class Source(Debuggable):
    """
    A state box that knows how to receive its value from somewhere
    """

    def __init__(self, upstream, shouldPull=(lambda: True), name = None, debug = False):
        """
        shouldPull should be a function that can be used to check whether there is a new value for this source
        available
        """
        Debuggable.__init__(self, name or ("->"+randomName(5)), debug)
        self.value = None
        self.upstream = upstream
        self.shouldPull = shouldPull
        self.subscribers = []

    def push(self, _ = None, value = None):
        """
        2nd argument is not used.
        Only pull a new value from upstream if it's fresh. By default always pull.
        If the value pulled is fresh push it to all subscribers
        """
        assert value == None # Shouldn't push values to a source?
        if self.shouldPull():
            old_val = self.value
            self.value = self.upstream()
            if self.value != old_val:
                for sub in self.subscribers:
                    sub.push(self,self.value)

    def pull(self):
        """
        Return the value that was last pulled from upstream.
        The value is only fresh if this operation was preceded by a push()
        """
        return self.value

    def addSubscriber(self, sub):
        self.subscribers.append(sub)


class Sink(Debuggable):
    """
    A signal sink. This class by itself should only be useful to make IO or trivial sinks.
    """

    def __init__(self, sources, downstream = (lambda x: None), name = None, debug = False):
        Debuggable.__init__(self, name or (randomName(5)+"->"), debug)
        self.subscriptions = {s: False for s in sources} # {source: have_latest}
        for s in sources:
            s.addSubscriber(self)
        self.downstream = downstream

    def push(self, source, value = None):
        assert source in self.subscriptions
        if not value is None:
            if self._debug: self.debug(self.getName()+": "+value)
            self.subscriptions[source] = True
            self.downstream(value)
        else:
            self.subscriptions[source] = False

    def pull(self):
        """
        One of the mechanisms by which values are propagated.
        The sink will pull a value from its source if needed and push it downstream.
        """
        for sub in self.subscriptions.iterkeys():
            if not self.subscriptions[sub]:
                self.downstream(sub.pull())
                self.subscriptions[sub] = True

class Transformer(Debuggable):
    """
    A (function) transformation between sources and sinks.
    """

    def __init__(self, sources, arg_assigns, transform, no_updates = None, name = None, debug = False):
        """
        Make a new Transformer with the provided sources as inputs.
        :type sources: list
        :type arg_assigns: dict
        :type transform: function
        :type no_updates: list
        :type name: str
        :type debug: bool
        """
        Debuggable.__init__(self, name or ("-"+randomName(5)+"-"), debug)
        self.value = None
        self.transform = transform
        self.setSources(sources)
        self.arg_assigns = arg_assigns
        self.no_updates = no_updates or []
        self.inputs = {arg_assigns[s.getName()]: None for s in sources}
        self.subscribers = []

    def setSources(self, sources):
        for s in sources:
            s.addSubscriber(self)
        self.subscriptions = {s: False for s in sources} # Value is whether transformer has the latest value

    def push(self, source, value = None):
        #assert source in self.subscriptions
        #^ The above is not always true if sources can change
        if source in self.subscriptions:
            if not value is None:
                self.subscriptions[source] = True
                input_name = self.getInputNameFor(source)
                if input_name:
                    self.inputs[input_name] = value
                    # Don't update value on sources in no_updates
                    # If there is only 1 input we now have all necessary info already
                    # (or only 1 input not in no_updates, and it's the one that's updated)
                    if (self.arg_assigns[source.getName()] not in self.no_updates) and \
                            (len(self.inputs) == 1 or all(self.subscriptions.values()) or
                            len(self.inputs) - len(self.no_updates) == 1):
                        self.makeValue()
                        for scriber in self.subscribers:
                            scriber.push(self, self.value)
            else:
                self.subscriptions[source] = False

    def pull(self):
        values_changed = False
        for scription in self.subscriptions.iterkeys():
            if not self.subscriptions[scription]: # If we don't have the latest value
                self.inputs[self.getInputNameFor(scription)] = scription.pull()
                self.subscriptions[scription] = True
                values_changed = True
        if values_changed:
            self.makeValue()
        return self.value

    def makeValue(self):
        self.value = self.transform(**self.inputs)
        if self._debug: self.debug(self.getName()+": "+self.value)
        self.subscriptions = {s : False for s in self.subscriptions.keys()}

    def getInputNameFor(self, source):
        if source.getName() in self.arg_assigns:
            return self.arg_assigns[source.getName()]
        else:
            return None

    def addSubscriber(self, sub):
        self.subscribers.append(sub)


class SignalBlock(object):
    """
    A SignalBlock is (in principle) a signal graph from a set of sources to a set of sinks.
    Because at the moment the connections are handled when setting up the transformers and sinks this is a rather
    empty class, but the plan is to add code to build these graphs from a description.
    Activating a process causes updated values to be sent to the sinks if necessary.
    """

    def __init__(self, sources, sinks = None):
        """
        Takes pre-setup sources and sinks.
        :type sources: list
        :type sinks: list
        """
        self.sources = sources
        self.sinks = sinks or []
        self.started = False
        self.finished = True

    def activate(self):
        self.finished = False
        for s in self.sources:
            s.push()
        for s in self.sources:
            if hasattr(s, "finished"):
                while not s.finished:
                    time.sleep(0.1)
        self.finished = True
        return [s.pull() for s in self.sinks]

    def build_block(self, description):
        """
        Takes a (json) dict description of a signal graph,
        constructs appropriate sources, transformers and sinks,
        puts the sources and sinks in a new SignalBlock.
        :param description: A dictionary with names as keys and objects with type and connections as values
        :return: A SignalBlock that follows the description given in description
        """
        pass

    def start(self):
        self.started = True
        parent = self
        def internal():
            while parent.started:
                if parent.finished:
                    parent.activate()
                else:
                    time.sleep(0.1)
        self.t = Thread(target=internal)
        self.t.daemon = True
        self.t.start()

    def stop(self):
        self.started = False
        while not self.finished:
            pass # Active wait until finished

class Accumulator(Transformer):
    """
    Accumulates values from another source up to a certain number, pushing them out as a list
    """
    def __init__(self, source, length, initial_list=None):
        self.accumulated = 0
        self.length = length
        self.collection = initial_list or []
        Transformer.__init__(self, [source], {source.getName(): 'm'}, self.accumulate)

    def accumulate(self, m):
        self.collection = self.collection[1:]+[m]
        self.accumulated += 1
        if self.accumulated == self.length:
            for scriber in self.subscribers:
                scriber.push(self, self.collection)
            self.accumulated = 0

    def pull(self):
        return self.collection

class ListSource(Source):
    def __init__(self, a_list, stop_callback = None):
        Source.__init__(self, self.next_value)
        self.list = a_list
        self.callback = stop_callback
        self.index = -1

    def next_value(self):
        self.index += 1
        if self.index == len(self.list):
            if not self.callback:
                self.index = 0 # By default the list just loops
            else:
                self.callback()
                return self.value
        return self.list[self.index]

class GenSource(Source):
    def __init__(self, a_gen, stop_callback = None):
        Source.__init__(self, self.next_value)
        self.callback = stop_callback
        self.gen = a_gen

    def next_value(self):
        try:
            return next(self.gen)
        except StopIteration as s:
            self.logger.debug(s)
            if self.callback: self.callback()
            return self.value

class CallbackSink(Sink):
    # This could probably be done with a regular Sink... Oh well.
    def __init__(self, sources, predicate, callback):
        Sink.__init__(self, sources)
        self.predicate = predicate
        self.callback = callback

    def push(self, source, value = None):
        if self.predicate(value):
            self.callback(value)
