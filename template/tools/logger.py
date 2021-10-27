import os

logger = None

class LogRecorder:
    def __init__(self, path, overwrite=True):
        self.path = path
        if  '/' in path:
            pos = path.rfind('/')
            self.dir, self.file = path[:pos], path[pos+1:]
            if not os.path.exists(self.dir):
                os.makedirs(self.path)
                print("create path: "+self.dir)
        if overwrite and os.path.exists(path):
            os.remove(path)
            print("remove old file: " + path)

    def info(self, str, addition_std = None, addition_log = None):
        # if addition_std is None:
        #     print(str)
        # else:
        #     print(str+addition_std)
        with open(self.path, "a") as file:
            if addition_log is None:
                file.write(str+"\n")
            else:
                file.write(str+addition_log+"\n")

def initialize_logger(save_path, overwrite=True):
    global logger
    logger = LogRecorder(save_path+"/log.txt", overwrite)

def initialize_test_logger(save_path, overwrite=True):
    global logger
    logger = LogRecorder(save_path+"/test_log.txt", overwrite)

def get_logger():
    return logger
