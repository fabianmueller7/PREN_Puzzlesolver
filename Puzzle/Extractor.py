import os

import cv2
import numpy as np

from Img.GreenScreen import remove_background
from Img.filters import export_contours_without_colormatching


PREPROCESS_DEBUG_MODE = 0  # set to 1 if you want debug images on disk


def show_image(img, name="image"):
    """Helper for quick visual debugging (only used if PREPROCESS_DEBUG_MODE == 1)."""
    import matplotlib.pyplot as plt

    plt.axis("off")
    plt.title(name)
    plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
    plt.show()


class Extractor:
    """
    Preprocessing + piece extraction for puzzle images.
    Pipeline idea:
      1) Optional green-screen removal
      2) Try simple Otsu threshold (fast path)
      3) If that fails, use robust preprocessing + watershed
      4) Filter contours by area and export pieces
    """

    def __init__(self, path, viewer=None, green_screen=False, factor=0.84):
        self.path = path
        self.viewer = viewer
        self.green_ = green_screen
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self.img_bw = None
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.temp_dir = os.environ.get("ZOLVER_TEMP_DIR", ".")
        self.factor = factor

        if self.green_:
            self._apply_green_screen()

        if self.img_bw is None:
            # default grayscale
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def log(self, *args):
        msg = " ".join(map(str, args))
        print(msg)
        if self.viewer:
            self.viewer.addLog(msg)

    def _save_temp(self, filename, img):
        try:
            cv2.imwrite(os.path.join(self.temp_dir, filename), img)
        except Exception as e:
            print("Failed to write temp image:", filename, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self):
        """
        Main entry point:
          - choose segmentation strategy
          - optional watershed separation
          - final contour filtering + export
        """
        self.log(">>> START preprocessing")

        used_simple = False
        backup_bw = None

        if not self.green_:
            used_simple = self.simple_piece_threshold(min_pieces=2, max_pieces=10)
            self.log("Using simple threshold segmentation (no watershed).")

        else:
            # green-screen: we already have a fairly clean mask, just clean & separate
            self._basic_morphology()
            backup_bw = self.img_bw.copy()

        if PREPROCESS_DEBUG_MODE:
            show_image(self.img_bw, "final_bw_before_fallback")

        # 2) contours from current mask
        contours, _ = cv2.findContours(self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.log("Found nb pieces:", len(contours))

        # 3) Fallback only if we did not use simple-threshold path
        if not used_simple and backup_bw is not None and len(contours) < 3:
            self.log("Too few pieces, using Otsu fallback segmentation.")
            self.img_bw = self._otsu_cleanup(backup_bw)
            contours, _ = cv2.findContours(
                self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            self.log("Found nb pieces after fallback:", len(contours))

        # 4) Filter contours by area
        contours = self._filter_contours_by_area(contours, min_ratio=0.33)
        self.log("Found nb pieces after removing small ones:", len(contours))

        # nothing useful -> abort
        if not contours:
            self.log("No valid pieces found.")
            return None

        # debug / viewer output
        self._save_temp("binarized_threshold_final.png", self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage(
                "Binarized threshold",
                os.path.join(self.temp_dir, "binarized_threshold_final.png"),
            )

        if PREPROCESS_DEBUG_MODE:
            debug = np.zeros_like(self.img_bw)
            cv2.drawContours(debug, contours, -1, 255, 1)
            self._save_temp("contours_debug.png", debug)
            show_image(debug, "final_contours")

        self.log(">>> START contour/corner detection")
        puzzle_pieces = export_contours_without_colormatching(
            self.img,
            self.img_bw,
            contours,
            os.path.join(self.temp_dir, "contours.png"),
            5,
            viewer=self.viewer,
            green=self.green_,
        )
        return puzzle_pieces

    # ------------------------------------------------------------------
    # Green-screen handling
    # ------------------------------------------------------------------

    def _apply_green_screen(self):
        """
        Downscale image, remove green background and read back as grayscale mask.
        """
        img = cv2.medianBlur(self.img, 5)

        # normalize width to ~640px
        scale = 640.0 / img.shape[1]
        self.log("Original shape:", img.shape, "| resizing with factor", scale)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self._save_temp("resized.png", img)
        remove_background(os.path.join(self.temp_dir, "resized.png"), factor=self.factor)

        # background-removed image should already be grayscale
        green_removed_path = os.path.join(self.temp_dir, "green_background_removed.png")
        self.img = img  # keep scaled color version
        self.img_bw = cv2.imread(green_removed_path, cv2.IMREAD_GRAYSCALE)

    # ------------------------------------------------------------------
    # Preprocessing strategies
    # ------------------------------------------------------------------

    def _basic_morphology(self):
        """
        Simple open/close on current self.img_bw.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        self.img_bw = bw

    def _otsu_cleanup(self, gray_or_bw):
        """
        Otsu binarization + light morphology cleanup.
        """
        if len(gray_or_bw.shape) > 2:
            gray = cv2.cvtColor(gray_or_bw, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_or_bw

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        if PREPROCESS_DEBUG_MODE:
            self._save_temp("fallback_bw.png", bw)

        return bw

    # ------------------------------------------------------------------
    # Simple threshold fast-path
    # ------------------------------------------------------------------

    def simple_piece_threshold(self, min_pieces=2, max_pieces=200):
        eps_mulitplicator = 0.0001 # Value from 0.0000.. to 0.15 ; Defines how round the borders can be (higher is more straight)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # --- 1. Reflection suppression ---
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Clip top highlights
        high = np.percentile(gray, 98)
        gray = np.clip(gray, 0, high).astype(np.uint8)

        # Optional gamma compression (further reduces bright glare)
        gray = np.power(gray / 255.0, 0.8)
        gray = (gray * 255).astype(np.uint8)

        # --- 2. Threshold ---
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- 3. Morphological smoothing ---
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Optional: Gaussian blur + rethreshold to smooth jaggies
        bw = cv2.GaussianBlur(bw, (5, 5), 0)
        _, bw = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY)

        self._save_temp("simple_bw_raw.png", bw)

        # --- 4. Contours ---
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(contours[0])
        area_thr = max_area * 0.05
        good_contours = [c for c in contours if cv2.contourArea(c) > area_thr]

        n = len(good_contours)
        self.log(f"simple_piece_threshold: {n} große Konturen gefunden")

        if not (min_pieces <= n <= max_pieces):
            return False

        # --- 5. Fill + contour smoothing ---
        filled = np.zeros_like(bw)
        cv2.drawContours(filled, good_contours, -1, 255, thickness=cv2.FILLED)

        # Light blur + rethreshold to smooth inside edges
        filled = cv2.GaussianBlur(filled, (5, 5), 0)
        _, filled = cv2.threshold(filled, 128, 255, cv2.THRESH_BINARY)

        # --- 6. Geometric smoothing of contours ---
        smooth_contours = []
        for c in good_contours:
            # Increase eps for smoother outline
            eps = eps_mulitplicator * cv2.arcLength(c, True)
            smooth = cv2.approxPolyDP(c, eps, True)
            smooth_contours.append(smooth)
        good_contours = smooth_contours

        # Redraw final mask from smoothed contours
        self.img_bw = np.zeros_like(bw)
        cv2.drawContours(self.img_bw, good_contours, -1, 255, thickness=cv2.FILLED)

        self._save_temp("simple_bw_filled.png", self.img_bw)
        return True


    # ------------------------------------------------------------------
    # Contour post-filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_contours_by_area(contours, min_ratio=0.33):
        """
        Keep only contours with area >= (min_ratio * max_area).
        """
        if not contours:
            return []

        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        thr = max_area * min_ratio
        return [c for c, a in zip(contours, areas) if a >= thr]
