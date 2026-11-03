"""Character pipeline: teacher -> checker -> judge -> clean chats -> SFT -> arms -> eval.

Most stage code is ported from private research repositories with its origin
noted at the top of each module. Nothing in this package downloads weights or
starts a job on import.
"""

__version__ = "0.2.0"
