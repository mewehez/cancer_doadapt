import time
import math


def as_minutes(s):
    """Formats seconds to min sec."""
    m = math.floor(s / 60)
    s -= m * 60
    return "{:3d}m {:5.2f}s".format(m, s)


def _get_bar(index, size, barwidth=40):
    perc = index / size
    progress = int(1000 * perc) % 10
    finished = index * barwidth // size
    remaining = barwidth - finished
    prog_str = ""
    if progress != 0:
        remaining -= 1
        prog_str = str(progress)
    bar = "#" * finished
    space = " " * remaining
    return "[{}]".format(bar + prog_str + space)


def _get_bar_stats(start, vals=None):
    now = time.time()
    ellapsed = now - start
    time_str = "time: {}".format(as_minutes(ellapsed))
    valstr = ""
    if vals is not None:
        for key, val in vals.items():
            valstr += "{}: {}|".format(key, val)

    return "[{}{}]".format(valstr, time_str)


def make_progressbar(size, name="", barwidth=40):
    index = 0
    maxsize = 0
    start = time.time()
    name = "" if name.strip() == "" else name + " "

    def progressbar(step=1, vals=None):
        nonlocal index, maxsize, start
        if index < size:
            index += step  # step forward
            perc = index / size

            # compute string for progress bar parts
            bar_prog = "[{}{:5.1f}%]".format(name, 100 * perc)
            bar_str = _get_bar(index, size, barwidth=barwidth)
            bar_stats = _get_bar_stats(start, vals=vals)
            line = bar_prog + bar_str + bar_stats

            # adjust line width
            linesize = len(line)
            if linesize >= maxsize:
                maxsize = linesize
            nspace = maxsize - linesize
            line += " " * nspace

            # print line
            print(line, end='\r', flush=True)
            # new line if end of progress
            if index == size:
                print("")
        pass

    return progressbar
