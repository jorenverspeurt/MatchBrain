from __future__ import print_function
import string
import random
import logging

__author__ = 'joren'

"""
Primitives for signal handling (FRP style, but less advanced?)
Groups of signals are threaded through Processes. Processes consist of a collection of Sources, Transformers and Sinks.
Sources pull values from a provided method and Sinks push them out.
Transformers take a list of sources (Sources or Transformers) as inputs and a transformation to produce 1 output.
A Process works by first activating all the Sources (propagating an "update" metasignal down the structure).
When this is completed the final values are pulled from the Sinks, causing computation requests to propagate up the
structure.
"""

def randomName(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

class Debuggable(object):
    """
    Things that have a name and a signal or value that is printed based on the debug parameter.
    """

    def __init__(self, name, debug):
        self._name = name
        self._debug = debug

    def getName(self):
        return self._name

    def debug(self, value = None):
        if value is None:
            return self._debug
        else:
            print(self._name + ": " + value)


class Source(Debuggable):
    """
    A state box that knows how to receive its value from somewhere
    """

    def __init__(self, upstream, shouldPull=(lambda: True), name = None, debug = False):
        """
        shouldPull should be a function that can be used to check whether there is a new value for this source
        available
        """
        Debuggable.__init__(self, name or ("<-"+randomName(5)), debug)
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

    def __init__(self, sources, downstream = (lambda x: print(x)), name = None, debug = False):
        Debuggable.__init__(self, name or ("->"+randomName(5)), debug)
        self.subscriptions = {s: True for s in sources} # {source: should_fetch}
        for s in sources:
            s.addSubscriber(self)
        self.downstream = downstream

    def push(self, source, value = None):
        assert source in self.subscriptions
        if not value is None:
            if self.debug():
                print(self.getName()+": "+value)
            self.subscriptions[source] = False
            self.downstream(value)
        else:
            self.subscriptions[source] = True

    def pull(self):
        """
        One of the mechanisms by which values are propagated.
        The sink will pull a value from its source if needed and push it downstream.
        """
        for sub in self.subscriptions.iterkeys():
            if self.subscriptions[sub]:
                self.downstream(sub.pull())
                self.subscriptions[sub] = False


class LogSink(Sink):
    """
    All this sink does is log the provided dicts to a data logger
    """

    def __init__(self, source, logger, level = logging.INFO):
        Sink.__init__(self, [source], (lambda m: logger.log(level, m)), logger.name, debug = False)


class Transformer(Debuggable):
    """
    A (function) transformation between sources and sinks.
    """

    def __init__(self, sources, arg_assigns, transform, name = None, debug = False):
        """
        Make a new Transformer with the provided sources as inputs.
        sources

        :type sources: list
        :type arg_assigns: dict
        :type transform: function
        :type name: str
        :type debug: bool
        """
        Debuggable.__init__(self, name or ("--"+randomName(5)), debug)
        self.value = None
        self.transform = transform
        for s in sources:
            s.addSubscriber(self)
        self.subscriptions = {s: True for s in sources}
        self.arg_assigns = arg_assigns
        self.inputs = {arg_assigns[s.getName()]: None for s in sources}
        self.subscribers = []

    def push(self, source, value = None):
        assert source in self.subscriptions
        if not value is None:
            self.subscriptions[source] = False
            self.inputs[self.getInputNameFor(source)] = value
            if len(self.inputs) == 1: # if there is only 1 input we now have all necessary info already
                self.makeValue()
                for scriber in self.subscribers:
                    scriber.push(self.value)
        else:
            self.subscriptions[source] = True

    def pull(self):
        for scription in self.subscriptions.iterkeys():
            if self.subscriptions[scription]:
                self.inputs[self.getInputNameFor(scription)] = scription.pull()
                self.subscriptions[scription] = False
        self.makeValue()
        return self.value

    def makeValue(self):
        self.value = self.transform(**self.inputs)
        if self.debug(): self.debug(self.getName()+": "+self.value)

    def getInputNameFor(self, source):
        return self.arg_assigns[source.getName()]

    def addSubscriber(self, sub):
        self.subscribers.append(sub)


class Process:
    """
    A Process is a signal graph from a set of sources to a set of sinks.
    Running a process causes updated values to be sent to the sinks if necessary.
    """

    def __init__(self, sources, connections, sinks):
        """
        :param sources: A dict of named sources
        :param connections: A graph representation: for every primary and secondary source the appropriate sink is
        specified in a dictionary
        :param sinks: A dict of named sinks
        """
        self.sources = sources.values()
        self.sinks = sinks.values()
        self.graph = []
        # If the sources are connected directly to sinks in the connection dict (why?) make trivial pipes
        for source in sources:
            for sconn in connections[source]:
                if sconn in sinks.iterkeys():
                    self.graph.append(Transformer(sources[source], (lambda x: x), sinks))
        #TODO complete graph building
