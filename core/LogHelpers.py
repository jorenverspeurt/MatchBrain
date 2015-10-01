__author__ = 'joren'

import logging
import json

from pythonjsonlogger import jsonlogger


class CustomFormatter(jsonlogger.JsonFormatter):
    def process_log_record(self, log_record):
        if 'name' in log_record: # then it's from our standard format
            ret = {std:log_record[std] for std in self._required_fields}
            paths = log_record['name'].split('.')
            paths.reverse()
            base = {other:log_record[other]
                    for other in log_record
                    if other not in self._required_fields}
            ret.update(reduce((lambda a,b: {b: a}), paths, base))
            return ret
        else:
            return log_record

class CustomHandler(logging.FileHandler):
    def __init__(self, filename, metadata):
        """
        :param filename: The filename to write the log to
        :param metadata: Whatever needs to be written as first object
        :return: None
        The whole reason for creating this class is to make it easy to log everything as a JSON
        array of objects. To do this it is necessary to write a [ with some first object before
        the rest and to write a final ] when closing.
        The json objects are all prefixed with a comma so they fit in the list.
        """
        logging.FileHandler.__init__(self,filename, mode='a', encoding='utf-8', delay=True)
        self.filename = filename
        defaultFormatter = CustomFormatter('%(msecs) %(name) %(message)')
        f = open(filename, mode='w')
        f.write('['+json.dumps(
            metadata,
            default=defaultFormatter.json_default,
            cls=defaultFormatter.json_encoder)+'\n')
        f.close()
        defaultFormatter.prefix = ','
        self.setFormatter(defaultFormatter)

    def close(self):
        f = open(self.filename, mode='a')
        f.write(']')




