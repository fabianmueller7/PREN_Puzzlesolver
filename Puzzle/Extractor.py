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
            # 1) simple fast-path for high-contrast images
            used_simple = self.simple_piece_threshold(min_pieces=2, max_pieces=200)
            if used_simple:
                self.log("Using simple threshold segmentation (no watershed).")
            else:
                self.log("Simple threshold not suitable, switching to advanced preprocessing.")
                self._advanced_preprocessing()
                backup_bw = self.img_bw.copy()
                self.separate_pieces()
        else:
            # green-screen: we already have a fairly clean mask, just clean & separate
            self._basic_morphology()
            backup_bw = self.img_bw.copy()
            self.separate_pieces()

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

    def _advanced_preprocessing(self):
        """
        Robust preprocessing for real puzzle images:
          - illumination normalization
          - CLAHE contrast
          - adaptive threshold
          - crack-closing morphology
        Sets self.img_bw (binary mask).
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # illumination normalization
        blur_bg = cv2.GaussianBlur(gray, (55, 55), 0)
        norm = cv2.addWeighted(gray, 1.5, blur_bg, -0.5, 0)
        norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(norm)

        # adaptive threshold
        bw = cv2.adaptiveThreshold(
            gray_eq,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            4,
        )

        # reinforce edges
        edges = cv2.Canny(gray_eq, 30, 90)
        bw = cv2.bitwise_or(bw, edges)

        # close cracks
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # fill internal holes
        contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(bw, [cnt], 0, 255, -1)

        # light smooth + clean
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)
        bw = cv2.GaussianBlur(bw, (3, 3), 0)
        _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)

        self.img_bw = bw

        if PREPROCESS_DEBUG_MODE:
            self._save_temp("pre_gray_eq.png", gray_eq)
            self._save_temp("adaptive_fixed.png", bw)

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
    # Watershed separation
    # ------------------------------------------------------------------

    def separate_pieces(self):
        """
        Watershed-based separation of touching pieces.
        Works on self.img_bw and updates it in-place.
        """
        self.log(">>> separate_pieces CALLED")

        FG_THRESH = 0.6
        DILATE_ITER = 1
        GAP_ERODE_ITER = 2
        DIST_SMOOTH = 3

        bw = self.img_bw.copy()
        if len(bw.shape) > 2:
            bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)

        orig_bw = bw.copy()

        # enlarge gaps between pieces
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.erode(bw, gap_kernel, iterations=GAP_ERODE_ITER)

        kernel_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_base, iterations=0)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_base, iterations=1)
        bw = cv2.GaussianBlur(bw, (1, 1), 0)

        # distance transform
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
        dist = cv2.GaussianBlur(dist, (DIST_SMOOTH, DIST_SMOOTH), 0)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        _, sure_fg = cv2.threshold(dist_norm, FG_THRESH, 1.0, cv2.THRESH_BINARY)
        sure_fg = (sure_fg * 255).astype(np.uint8)

        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sure_bg = cv2.dilate(bw, kernel_bg, iterations=DILATE_ITER)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        img_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        separated = np.zeros_like(bw)
        separated[markers > 1] = 255

        # restore original shape while preserving separation
        separated = cv2.bitwise_and(separated, orig_bw)

        # remove thin spurs + small blobs
        separated = self._remove_spurs_and_small(separated)

        self.img_bw = separated

        # debug visualization
        if PREPROCESS_DEBUG_MODE:
            debug = np.zeros_like(img_color)
            for label in np.unique(markers):
                if label <= 1:
                    continue
                mask = np.uint8(markers == label) * 255
                color = tuple(np.random.randint(0, 255, 3).tolist())
                debug[mask == 255] = color

            self._save_temp("watershed_param.png", separated)
            self._save_temp("watershed_param_debug.png", debug)

    def _remove_spurs_and_small(self, binary):
        """
        Removes small protrusions and small components (relative to the largest).
        """
        spur_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pruned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, spur_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            pruned, connectivity=8
        )
        if num_labels <= 1:
            return pruned

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        MIN_REL_AREA = 0.02
        min_keep_area = int(MIN_REL_AREA * max_area)

        cleaned = np.zeros_like(pruned)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_keep_area:
                cleaned[labels == label] = 255

        # light smoothing
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_final, iterations=1)
        return cleaned

    # ------------------------------------------------------------------
    # Simple threshold fast-path
    # ------------------------------------------------------------------

    def simple_piece_threshold(self, min_pieces=2, max_pieces=200):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu + invert
        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  kernel, iterations=1)

        self._save_temp("simple_bw_raw.png", bw)

        # Konturen suchen (nur äußere!)
        contours, _ = cv2.findContours(
            bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False

        # Nach Fläche sortieren
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(contours[0])
        area_thr = max_area * 0.05  # 5 % der größten Kontur
        good_contours = [c for c in contours if cv2.contourArea(c) > area_thr]

        n = len(good_contours)
        self.log(f"simple_piece_threshold: {n} große Konturen gefunden")

        if not (min_pieces <= n <= max_pieces):
            return False

        # ⬇️ WICHTIG: jedes Teil komplett ausfüllen
        filled = np.zeros_like(bw)
        cv2.drawContours(filled, good_contours, -1, 255, thickness=cv2.FILLED)

        self._save_temp("simple_bw_filled.png", filled)
        self.img_bw = filled
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
