import argparse
import glob
import os
import sys


def _setup_debug_dir():
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    for f in glob.glob(os.path.join(debug_dir, "*")):
        os.remove(f)
    os.environ["ZOLVER_TEMP_DIR"] = debug_dir
    print(f"Debug output directory: {debug_dir}")
    return debug_dir


def main():
    parser = argparse.ArgumentParser(description="PREN Puzzlesolver")
    parser.add_argument("--no-gui", action="store_true", help="run headless (no GUI)")
    parser.add_argument("--image", type=str, help="input image (required with --no-gui)")
    parser.add_argument("--green-screen", action="store_true", help="enable green background removal")
    parser.add_argument("--profile", action="store_true", help="enable cProfile (only with --no-gui)")
    args = parser.parse_args()

    _setup_debug_dir()

    if args.no_gui:
        if not args.image:
            parser.error("--image is required when using --no-gui")
        import multiprocessing as mp
        mp.set_start_method("fork")
        from solver.Puzzle.Puzzle import Puzzle
        if args.profile:
            import cProfile
            import pstats
            import io
            from pstats import SortKey
            with cProfile.Profile() as pr:
                Puzzle(args.image, green_screen=args.green_screen).solve_puzzle()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE).print_stats(50)
            print(s.getvalue())
        else:
            Puzzle(args.image, green_screen=args.green_screen).solve_puzzle()
    else:
        from PyQt5.QtWidgets import QApplication
        from solver.GUI.Viewer import Viewer
        app = QApplication(sys.argv)
        viewer = Viewer()
        viewer.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
