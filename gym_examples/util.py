import logging
#  from https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
class DuplicateFilter(logging.Filter):
    def __init__(self):
        self.count = 0

    def filter(self, record):
        # add other fields if you need more granular comparison, depends on your app
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            if self.count > 0:
                record.msg = f"{record.msg} repeated {self.count} times"
            self.last_log = current_log
            self.count = 0
            return True
        self.count = self.count + 1
        return False