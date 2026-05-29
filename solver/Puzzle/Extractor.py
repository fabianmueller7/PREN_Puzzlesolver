import os

import cv2
import numpy as np
from .. import config

from ..Img.GreenScreen import remove_background
from ..Img.filters import export_contours_without_colormatching

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
            config.save_debug_img(os.path.join(self.temp_dir, filename), img)
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
        used_visual = False
        backup_bw = None

        if not self.green_:
            used_visual = self.white_bg_threshold(min_pieces=2, max_pieces=10)
            used_simple = used_visual
            if used_visual:
                self.log("Using white-background segmentation.")
            else:
                used_simple = self.simple_piece_threshold(min_pieces=2, max_pieces=10)
                self.log("Using simple threshold segmentation (no watershed).")

        else:
            # green-screen: we already have a fairly clean mask, just clean & separate
            self._basic_morphology()
            backup_bw = self.img_bw.copy()

        if config.DEBUG_SHOW_DIAGRAMS == 1:
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

        if config.DEBUG_FILE_OUTPUT == 1:
            debug = np.zeros_like(self.img_bw)
            cv2.drawContours(debug, contours, -1, 255, 1)
            self._save_temp("contours_debug.png", debug)
            if config.DEBUG_SHOW_DIAGRAMS == 1:
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

        if used_visual and puzzle_pieces:
            self.log("[VISUAL CENTER] Centers detected via white-background method:")
            for i, p in enumerate(puzzle_pieces):
                if p.img_centroid is not None:
                    self.log(f"  piece {i}: cx={p.img_centroid[0]:.1f} px, cy={p.img_centroid[1]:.1f} px")
                else:
                    self.log(f"  piece {i}: centroid not available")

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

        if config.DEBUG_FILE_OUTPUT == 1:
            self._save_temp("fallback_bw.png", bw)

        return bw

    # ------------------------------------------------------------------
    # White-background fast-path (production)
    # ------------------------------------------------------------------

    def white_bg_threshold(self, min_pieces=2, max_pieces=200):
        """Detect pieces by finding non-white regions.

        Works well in production where the gamefield background is white and
        the LED illuminates it strongly.  Pieces appear as dark/coloured blobs
        against the bright white surface.

        Returns True and sets self.img_bw when it finds a plausible piece mask.
        """
        print("[white_bg] attempting visual dark-piece detection...")

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        dark_pct = 100.0 * np.sum(gray < 80) / gray.size
        print(f"[white_bg] dark pixel coverage (V<80): {dark_pct:.1f}%")

        # Pieces are very dark/black against a bright background
        _, piece_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove small salt-and-pepper noise
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, k_open, iterations=1)

        # Close small gaps inside pieces (tabs, text, texture)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[white_bg] FAILED — no contours after morphology")
            return False

        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        good = [c for c, a in zip(contours, areas) if a >= max_area * 0.05]

        n = len(good)
        print(f"[white_bg] {n} piece candidates (need {min_pieces}–{max_pieces})")

        if not (min_pieces <= n <= max_pieces):
            print(f"[white_bg] FAILED — piece count {n} outside [{min_pieces}, {max_pieces}]")
            return False

        filled = np.zeros_like(piece_mask)
        cv2.drawContours(filled, good, -1, 255, thickness=cv2.FILLED)

        self.img_bw = filled
        self._save_temp("white_bg_filled.png", self.img_bw)
        print(f"[white_bg] SUCCESS — {n} pieces detected visually")
        return True

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

        self._save_temp("debug_output/simple_bw_raw.png", bw)

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

        self._save_temp("debug_output/simple_bw_filled.png", self.img_bw)
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
