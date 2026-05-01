#! /bin/env python3

"""
This script is used to parse the output from the lua script in tshark.  The
benefit of this is that we can filter out any unexpected output from tshark
and we can combine the information over time.  This could have been done in
lua, but python is a much nicer language.

:author: Nick Jones
:date: 1/31/2016
"""

from collections import defaultdict, namedtuple
import fileinput
from datetime import datetime, timedelta
import sys

Beacon = namedtuple("Beacon", "bssid dbm")


def split_line(line):
    parts = line.strip().split("\t")
    if len(parts) != 2 or "" in parts or "nil" in parts:
        return None
    return Beacon(bssid=parts[0], dbm=int(parts[1]))


def print_states(states):
    for bssid, rssi in states.items():
        sys.stdout.write("%s %d\n" % (bssid, rssi))
    sys.stdout.write("-fingerprint-\n")
    sys.stdout.flush()


def main():
    """
    The way this works is by keeping a view of the world and a view
    within an interval.  If multiple beacons from a single AP are
    seen within the interval they are combined using a weighted average.
    If an access point does not appear in the interval, but we expected it,
    it gets penalized harshly.  APs that fall below -10dbm are removed from
    the view of the world.  At the end of each interval, the view of the
    world is written to stdout.
    """
    states = defaultdict(int)  # view of world over multiple states
    int_start = datetime.now()  # datetime of interval start
    int_states = defaultdict(int)  # states seen in interval
    for line in fileinput.input():
        b = split_line(line)
        if b is None:
            continue
        s = int_states[b.bssid]
        if s == 0:
            int_states[b.bssid] = b.dbm
        else:
            int_states[b.bssid] = b.dbm * .8 + s * .2

        if int_start + timedelta(milliseconds=1200) < datetime.now():
            for key, rssi in int_states.items():
                states[key] = rssi
            for key, rssi in states.items():
                if key not in int_states:
                    states[key] = states[key] / 2.0
            states = {key: states[key] for key in states
                      if abs(states[key]) > 10}
            print_states(states)
            int_start = datetime.now()
            int_states = defaultdict(int)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

