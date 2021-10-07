"""
A mock resource.py file to make SMAC work on Windows (not having the resource package)
"""

RUSAGE_CHILDREN = 0
RUSAGE_SELF = 0
RLIMIT_STACK = 0

def getrusage(*vargs, **kwargs):
    pass

def setrlimit(*vargs, **kwargs):
    pass