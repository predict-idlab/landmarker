"""Converts image (jpg or png) to DICOM as secondary capture, with DICOM tags.

Source: https://docs.md.ai/libraries/python/guides-convert-dcm/
"""

import os
import glob

import pydicom  # type: ignore
from tqdm import tqdm  # type: ignore
import cv2

from landmarker.utils.utils import get_paths


def append_files_extension(folder: str, extension: str) -> None:
    """Append file extension to all files and sub folders in folder.

    Args:
        folder (str): folder
        extension (str): file extension
    """
    for path in glob.glob(folder + r'/**/*', recursive=True):
        if os.path.isfile(path):
            os.rename(path, path + extension)


def convert_all_dcm_png(source_folder: str, target_folder: str) -> None:
    """Covert all dicoms in folder to png

    Args:
        source_folder (String): source folder path name
        target_folder (String): target folder path name
    """
    source_path_names = get_paths(source_folder, "dcm")
    files_no_pixel_data = []
    files_error = []
    for name in tqdm(source_path_names):
        ds = pydicom.dcmread(name)
        if "PixelData" not in ds:
            files_no_pixel_data.append(name)
            continue
        try:
            img = ds.pixel_array
        except:  # noqa: E722
            files_error.append(name)
            continue
        target_path = name.replace(".dcm", ".png")
        target_path = target_path.replace(source_folder, target_folder)
        if os.path.exists(os.path.dirname(target_path)) is False:
            os.makedirs(os.path.dirname(target_path))
        cv2.imwrite(target_path, img)
        print("Written shape: ", img.shape)
        print("Read shape: ", cv2.imread(target_path).shape)
    print("These files where not converted due to no pixel data: ")
    for name in files_no_pixel_data:
        print(name)
    print("These files where not converted due to no pixel data: ")
    for name in files_error:
        print(name)


# def from_jpeg(jpeg_fp, dicom_dir, dicom_tags={}):
#     """Converts JPEG to DICOM as secondary capture, with minimal DICOM tags.
#     If JPEG mode is RGBA or CMYK, we must first convert to RGB since these
#     photometric interpretations have been retired in the DICOM standard:
#     http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html.
#     Returns: path to DICOM tempfile
#     """
#     # If JPEG is RGBA/CMYK mode, convert to RGB mode first.
#     im = Image.open(jpeg_fp)
#     if im.mode in ("RGBA", "CMYK"):
#         im2 = im.convert("RGB")
#         im2.save(jpeg_fp)
#         im2.close()
#     im.close()

#     dicom_fp = os.path.join(dicom_dir, f"{str(uuid.uuid4())}.dcm")
#     cmd = ["img2dcm", jpeg_fp, dicom_fp]
#     for key, value in dicom_tags.items():
#         cmd.extend(["-k", f"{key}={value}"])
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                    check=True)
#     return dicom_fp


# def from_png(png_fp, dicom_dir, dicom_tags={}):
#     """Converts PNG to DICOM as secondary capture, with minimal DICOM tags.
#     Returns: path to DICOM tempfile
#     """
#     jpeg_fp = os.path.join(dicom_dir, f"{str(uuid.uuid4())}.jpg")
#     cmd = ["convert", png_fp, jpeg_fp]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                    check=True)
#     return from_jpeg(jpeg_fp, dicom_tags=dicom_tags)
