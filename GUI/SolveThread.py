# GUI/SolveThread.py
from PyQt5.QtCore import QThread, pyqtSignal
from Puzzle.Puzzle import Puzzle  # adjust import path to your project


class SolveThread(QThread):
    """Thread used to launch the puzzle solving"""

    # args is the tuple sent from Puzzle.log(*args)
    logAdded = pyqtSignal(object)
    # name, fileName, display, addMenu
    imageAdded = pyqtSignal(str, str, bool, bool)

    def __init__(self, path, viewer, green_screen=False):
        super().__init__(viewer)  # parent = viewer
        self.path = path
        self.viewer = viewer
        self.green_screen = green_screen

    def run(self):
        # Use a proxy instead of the real Viewer inside the worker thread
        proxy = ViewerProxy(self)
        puzzle = Puzzle(self.path, viewer=proxy, green_screen=self.green_screen)
        puzzle.solve_puzzle()


class ViewerProxy:
    """
    Passed to Puzzle/Extractor instead of the real Viewer.
    Only emits signals on SolveThread; no direct GUI calls from worker.
    """

    def __init__(self, thread: SolveThread):
        self._thread = thread

    def addLog(self, args):
        # args is (arg1, arg2, ...)
        self._thread.logAdded.emit(args)

    def addImage(self, name, fileName, display=True, addMenu=False):
        self._thread.imageAdded.emit(name, fileName, display, addMenu)
