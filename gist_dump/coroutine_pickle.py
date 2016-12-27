# Author: Kyle Kastner
# License: BSD 3-Clause
import time
import os
try:
    import Queue
except ImportError:
    import queue as Queue
from threading import Thread
# Twisted and lifted from
# http://www.dabeaz.com/coroutines/Coroutines.pdf

def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start

@coroutine
def threaded_writer():
    messages = Queue.Queue()
    def run_thread():
        while True:
            item = messages.get()
            if item is GeneratorExit:
                return
            else:
                save_path, to_pickle = item
                # simulate slow pickle to nfs
                time.sleep(5)
                print("Simulated write complete", save_path, to_pickle)
    Thread(target=run_thread).start()
    try:
        while True:
            item = (yield)
            messages.put(item)
    except GeneratorExit:
        messages.put(GeneratorExit)


pp = threaded_writer()
start = time.time()
save_path = os.path.join("blah", "tmp.pkl")
to_pickle = {"stuff": [1, 2, 3]}
pp.send((save_path, to_pickle))
time.sleep(1)
mid = time.time()
pp.send((save_path, to_pickle))
time.sleep(1)
end = time.time()
print(mid - start)
print(end - mid)
print(end - start)
pp.close()
