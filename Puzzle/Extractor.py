import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Img.GreenScreen import remove_background
from Img.filters import export_contours, export_contours_without_colormatching

PREPROCESS_DEBUG_MODE = 0


def show_image(img, ind=None, name="image", show=True):
    """Helper used for matplotlib image display"""
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()


def show_contours(contours, imgRef):
    """Helper used for matplotlib contours display"""
    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "cont.png"), whiteImg)

class Extractor:
    """
    Class used for preprocessing and pieces extraction
    """

    def __init__(self, path, viewer=None, green_screen=False, factor=0.84):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if green_screen:
            self.img = cv2.medianBlur(self.img, 5)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print("Resizing with factor", divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), self.img)
            remove_background(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), factor=factor)
            self.img_bw = cv2.imread(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "green_background_removed.png"), cv2.IMREAD_GRAYSCALE
            )
            # rescale self.img and self.img_bw to 640
        else:
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.viewer = viewer
        self.green_ = green_screen
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def log(self, *args):
        """Helper function to log informations to the GUI"""
        print(" ".join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def extract(self):
        """
        Perform the preprocessing of the image and call functions to extract
        informations of the pieces.
        """

        kernel = np.ones((3, 3), np.uint8)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"))

        ### Implementation of random functions, actual preprocessing is down below

        def fill_holes():
            """filling contours found (and thus potentially holes in pieces)"""

            contour, _ = cv2.findContours(
                self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        #def generated_preprocesing():
            #ret, self.img_bw = cv2.threshold(
                #self.img_bw, 254, 255, cv2.THRESH_BINARY_INV
            #)
            #cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "otsu_binarized.png"), self.img_bw)
            #self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            #self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        def generated_preprocesing():
            """Robust preprocessing for real puzzle images with crack-free contours."""

            import cv2, numpy as np, os

            # Grayscale conversion
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # Illumination normalization (removes uneven lighting)
            blur_bg = cv2.GaussianBlur(gray, (55, 55), 0)
            norm = cv2.addWeighted(gray, 1.5, blur_bg, -0.5, 0)
            norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

            # Gentle contrast boost
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(norm.astype(np.uint8))

            # Adaptive threshold with slightly smaller block size
            self.img_bw = cv2.adaptiveThreshold(
                gray_eq,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                4
            )

            # Add edge reinforcement
            edges = cv2.Canny(gray_eq, 30, 90)
            self.img_bw = cv2.bitwise_or(self.img_bw, edges)

            # Close cracks & fill gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel_close, iterations=3)

            # Fill internal holes completely
            contours, _ = cv2.findContours(self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

            # Smooth & denoise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel_open, iterations=1)
            self.img_bw = cv2.GaussianBlur(self.img_bw, (3, 3), 0)

            # Optional: final threshold to clean up blur residues
            _, self.img_bw = cv2.threshold(self.img_bw, 127, 255, cv2.THRESH_BINARY)

            # Save intermediate debug images (optional)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "pre_gray_eq.png"), gray_eq)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "pre_norm.png"), norm)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "adaptive_fixed.png"), self.img_bw)

        def real_preprocessing():
            """Apply morphological operations on base image."""
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        ### PREPROCESSING: starts there

        # With this we apply morphologic operations (CLOSE, OPEN and GRADIENT)
        if not self.green_:
            generated_preprocesing()
            self.separate_pieces()
        else:
            real_preprocessing()
            self.separate_pieces()
        # These prints are activated only if the PREPROCESS_DEBUG_MODE variable at the top is set to 1
        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        # With this we fill the holes in every contours, to make sure there is no fragments inside the pieces
        #if not self.green_:
        #    fill_holes()

        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage(
                "Binarized treshold", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png")
            )

        contours, hier = cv2.findContours(
            self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.log("Found nb pieces: " + str(len(contours)))

        # With this we can manually set the maximum number of pieces manually, or we try to guess their number
        # to guess it, we only keep the contours big enough
        nb_pieces = None

        # TEMPORARY TO AVOID DEBUG ORGINAL:
        if len(sys.argv) < 0:
            # Number of pieces specified by user
            nb_pieces = int(sys.argv[2])
            contours = sorted(
                np.array(contours), key=lambda x: x.shape[0], reverse=True
            )[:nb_pieces]
            self.log("Found nb pieces after manual setting: " + str(len(contours)))
        else:
            # Try to remove useless contours
            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            max = contours[1].shape[0]
            contours = [elt for elt in contours if elt.shape[0] > max / 3]
            self.log("Found nb pieces after removing bad ones: " + str(len(contours)))

        if PREPROCESS_DEBUG_MODE == 1:
            show_contours(contours, self.img_bw)  # final contours

        ### PREPROCESSING: the end

        # In case with fail to find the pieces, we fill some holes and then try again
        # while True: # TODO Add this at the end of the project, it is a fallback tactic

        self.log(">>> START contour/corner detection")
        puzzle_pieces = export_contours_without_colormatching(
            self.img,
            self.img_bw,
            contours,
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "contours.png"),
            5,
            viewer=self.viewer,
            green=self.green_,
        )
        if puzzle_pieces is None:
            # Export contours error
            return None
        return puzzle_pieces

    def separate_pieces(self):
        import cv2, numpy as np, os
        print(">>> separate_pieces CALLED")

        FG_THRESH  = 0.6
        DILATE_ITER = 1
        CLOSE_ITER  = 0
        OPEN_ITER   = 1
        BLUR_RADIUS = 1
        DIST_SMOOTH = 3
        POST_CLOSE  = 0
        POST_OPEN   = 1
        GAP_ERODE_ITER = 2

        bw = self.img_bw.copy()
        if len(bw.shape) > 2:
            bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)

        # 🔴 Save original silhouette so we can restore real shape later
        orig_bw = bw.copy()

        # 🔴 ENLARGE GAPS: small erosion makes narrow black gaps bigger,
        #    so watershed cannot accidentally connect neighboring pieces.

        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.erode(bw, gap_kernel, iterations=GAP_ERODE_ITER)

        kernel_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_base, iterations=CLOSE_ITER)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  kernel_base, iterations=OPEN_ITER)
        bw = cv2.GaussianBlur(bw, (BLUR_RADIUS, BLUR_RADIUS), 0)

        # === distance transform ===
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

        # 🔴 Clip back to the ORIGINAL mask, not the eroded one
        #    -> shapes stay close to original, but separation comes from the erosion.
        separated = cv2.bitwise_and(separated, orig_bw)

        # (optional) tiny clean-up, you already had POST_OPEN etc:
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        separated = cv2.morphologyEx(separated, cv2.MORPH_OPEN, kernel_final,
                                     iterations=POST_OPEN)

        # debug images etc. unchanged ...
        out_dir = os.environ["ZOLVER_TEMP_DIR"]
        debug = np.zeros_like(img_color)
        for label in np.unique(markers):
            if label <= 1:
                continue
            mask = np.uint8(markers == label) * 255
            color = tuple(np.random.randint(0, 255, 3).tolist())
            debug[mask == 255] = color

        cv2.imwrite(os.path.join(out_dir, "watershed_param.png"), separated)
        cv2.imwrite(os.path.join(out_dir, "watershed_param_debug.png"), debug)

        self.img_bw = separated


        # 2) Remove thin “spurs” and small blobs
        def remove_spurs(binary):
            """
            Removes small protrusions and noise attached via thin connections.
            """
            # a) Cut off 1px-thick connections
            spur_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pruned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, spur_kernel, iterations=1)

            # b) Remove tiny islands that got detached
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                pruned, connectivity=8
            )

            # stats[:, cv2.CC_STAT_AREA] -> area per label
            areas = stats[1:, cv2.CC_STAT_AREA]  # ignore background label 0
            if len(areas) == 0:
                return pruned

            max_area = areas.max()
            # everything smaller than X% of the biggest component is treated as junk
            MIN_REL_AREA = 0.02  # 2% – tweak if needed
            min_keep_area = int(MIN_REL_AREA * max_area)

            cleaned = np.zeros_like(pruned)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_keep_area:
                    cleaned[labels == label] = 255

            return cleaned

        separated = remove_spurs(separated)

        # 3) Optional very light smoothing / denoising
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        separated = cv2.morphologyEx(separated, cv2.MORPH_OPEN, kernel_final,
                                     iterations=POST_OPEN)

        # === DEBUG VISUALIZATION ===
        debug = np.zeros_like(img_color)
        for label in np.unique(markers):
            if label <= 1:
                continue
            mask = np.uint8(markers == label) * 255
            color = tuple(np.random.randint(0, 255, 3).tolist())
            debug[mask == 255] = color

        out_dir = os.environ["ZOLVER_TEMP_DIR"]
        cv2.imwrite(os.path.join(out_dir, "watershed_param.png"), separated)
        cv2.imwrite(os.path.join(out_dir, "watershed_param_debug.png"), debug)

        self.img_bw = separated
