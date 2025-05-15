# Note: code written with the help of Open AI' ChatGPT o3 model
import fitz  # PyMuPDF
import cv2   # OpenCV
import numpy as np
import os
import shutil

# --- Parameters for Image Processing ---
# For line detection using morphological operations:
HORIZONTAL_KERNEL_LENGTH_RATIO = 0.05 # Ratio of page width, e.g., 0.05 means kernel is 5% of image width.
                                     # Increase if lines are very long and thick, decrease for shorter/thinner.
MIN_LINE_WIDTH_RATIO = 0.7         # Minimum width of a detected line contour relative to page width.
MAX_LINE_HEIGHT = 15               # Maximum height of a detected line contour in pixels.
                                     # Adjust based on line thickness in your scan and image DPI.

#MIN_ANSWER_HEIGHT was found experimentally ensuring bottom of page is not detected as problem.
MIN_ANSWER_HEIGHT = 250 # in pixels. 
EROSION_ITERATIONS = 1
DILATION_ITERATIONS = 1

# For output images
OUTPUT_IMAGE_DPI = 200  # DPI for rendering PDF page to an image. Higher DPI = larger image, better detail.
DEBUG_IMAGE_OUTPUT = False # Set to True to save images with detected lines drawn on them.
# --- End of adjustable parameters ---

IMAGES_FOLDER = "problem_images/"

# --- Helper functions --- #
def _debug_print(*args, **kwargs):
    """
    Custom print function that only prints if DEBUG_IMAGE_OUTPUT is True.
    Accepts arguments just like the built-in print().
    """
    if DEBUG_IMAGE_OUTPUT:
        print(*args, **kwargs)


def _convert_pixmap_to_opencv_img(pix):
    """Converts a PyMuPDF Pixmap to an OpenCV BGR image."""
    # pix.samples is in RGB or RGBA format
    if pix.alpha: # RGBA
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) # Convert RGBA to BGR
    else: # RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Already BGR if samples are in BGR order (common), or RGB if in RGB order.
        # PyMuPDF samples are typically RGB, so convert to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def _detect_horizontal_lines_in_image(image_cv):
    """
    Detects horizontal lines in an OpenCV image.
    Returns a sorted list of the top y-coordinates of detected horizontal lines.
    """
    if image_cv is None:
        _debug_print("  [Line Detection] Error: Input image is None.")
        return []
        
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 2. Apply adaptive thresholding to binarize the image
    # This can be better than simple thresholding for varying lighting/contrast in scans.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) # Block size 11, C 2
    # Alternatively, Otsu's thresholding after Gaussian blur:
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 3. Define a horizontal kernel
    image_height, image_width = binary.shape[:2]
    kernel_length = int(image_width * HORIZONTAL_KERNEL_LENGTH_RATIO)
    if kernel_length < 5: kernel_length = 5 # Minimum kernel length
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # 4. Apply morphological opening (erosion followed by dilation) to isolate horizontal lines
    # Erosion will remove vertical lines and thin elements.
    eroded = cv2.erode(binary, horizontal_kernel, iterations=EROSION_ITERATIONS)
    # Dilation will restore the horizontal lines that survived erosion.
    dilated = cv2.dilate(eroded, horizontal_kernel, iterations=DILATION_ITERATIONS)
    
    if DEBUG_IMAGE_OUTPUT:
        cv2.imwrite(f"debug_binary_page_{page_num_global}.png", binary)
        cv2.imwrite(f"debug_dilated_lines_page_{page_num_global}.png", dilated)

    # 5. Find contours of these horizontal lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_y_coords = []
    min_line_actual_width = image_width * MIN_LINE_WIDTH_RATIO

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_line_actual_width and h <= MAX_LINE_HEIGHT:
            # It's a contour that is wide enough and not too tall.
            # Add the top y-coordinate of the bounding box of the line.
            line_y_coords.append(y)
            if DEBUG_IMAGE_OUTPUT:
                cv2.rectangle(image_cv, (x,y), (x+w, y+h), (0,255,0), 2) # Draw green box on original

    if DEBUG_IMAGE_OUTPUT and contours:
         cv2.imwrite(f"debug_detected_lines_on_page_{page_num_global}.png", image_cv)


    return sorted(list(set(line_y_coords)))

def _strip_pdf_extension_os(filename):
    """
    Strips the .pdf extension from a filename, case-insensitively.
    Returns the name without the extension if it was .pdf, otherwise the original name.
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() == ".pdf":
        return name
    else:
        # If it's not a .pdf extension, return the original filename
        # (or you could choose to return name + ext if you always want to strip any ext)
        return filename

page_num_global = 0 # For unique debug image names

def _split_pdf_by_detected_lines(pdf_path, output_dir):
    """
    Renders PDF pages as images, detects horizontal lines, and saves segments.
    """
    global page_num_global

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        doc = fitz.open(pdf_path)
        print(f"Successfully opened PDF: {pdf_path}. Pages: {len(doc)}")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    overall_problem_idx = 1

    for page_idx in range(len(doc)):
        page_num_global = page_idx + 1 # For debug file naming
        page = doc.load_page(page_idx)
        if DEBUG_IMAGE_OUTPUT: _debug_print(f"\nProcessing Page {page_num_global}...") 

        # 1. Render page to an image (Pixmap)
        try:
            pix = page.get_pixmap(dpi=OUTPUT_IMAGE_DPI)
        except Exception as e:
            print(f"  Error rendering page {page_num_global} to pixmap: {e}")
            continue
            
        # 2. Convert Pixmap to OpenCV image format
        page_image_cv = _convert_pixmap_to_opencv_img(pix)
        if page_image_cv is None:
            print(f"  Error converting pixmap for page {page_num_global} to OpenCV image.")
            continue
        
        page_height, page_width = page_image_cv.shape[:2]
        _debug_print(f"  Page image dimensions: {page_width}x{page_height} (WxH) at {OUTPUT_IMAGE_DPI} DPI")

        # 3. Detect horizontal lines in the image
        # Create a copy for drawing debug lines, if enabled, to avoid modifying the original
        debug_image_copy = page_image_cv.copy() if DEBUG_IMAGE_OUTPUT else page_image_cv
        line_y_coords = _detect_horizontal_lines_in_image(debug_image_copy) # Pass copy for debugging
        _debug_print(f"  Found {len(line_y_coords)} potential horizontal lines at y-coordinates: {line_y_coords}")

        # 4. Define slices based on line coordinates
        slice_y_boundaries = [0] + line_y_coords + [page_height] # Add page top and bottom
        slice_y_boundaries = sorted(list(set(slice_y_boundaries))) # Ensure uniqueness and order

        _debug_print(f"  Defining slices with Y boundaries: {slice_y_boundaries}")

        for i in range(len(slice_y_boundaries) - 1):
            y_start = slice_y_boundaries[i]
            y_end = slice_y_boundaries[i+1]

            # Add a small gap so the line itself is not included in either slice.
            # Or, decide if line belongs to strip above or below.
            # Current: y_start is top of line (or page_top), y_end is top of next line (or page_bottom)
            # So, the slice is [y_start, y_end-1]
            
            # Ensure the slice has a minimum height
            if y_end > y_start + MIN_ANSWER_HEIGHT: # Using MIN_ANSWER_HEIGHT as min slice height
                # Crop the slice from the original page image
                # OpenCV slicing is [y_start:y_end, x_start:x_end]
                image_slice = page_image_cv[y_start:y_end, 0:page_width]
                
                output_filename = os.path.join(output_dir, f"problem_{overall_problem_idx}.jpg")
                _debug_print(f" saving image at: {output_filename}")
                try:
                    cv2.imwrite(output_filename, image_slice)
                    _debug_print(f"    Saved slice: {output_filename} (Y-range: {y_start}-{y_end})")
                    overall_problem_idx += 1
                except Exception as e:
                    print(f"    Error saving slice {output_filename}: {e}")
            else:
                _debug_print(f"    Skipping slice from Y {y_start} to {y_end}: too short (height {y_end-y_start})")
    
    doc.close()
    _debug_print(f"\nProcessing complete. {overall_problem_idx -1} slices saved.")


def _process_exam(exam_filename):
    assert exam_filename, "Error: An exam filename must be specified."
    pdf_file_path = exam_filename
    output_slices_dir = IMAGES_FOLDER

    _debug_print("Starting image-based slicing script...")
    _debug_print(f"Debug image output is: {'Enabled' if DEBUG_IMAGE_OUTPUT else 'Disabled'}")
    _debug_print(f"Parameters for line detection:")
    _debug_print(f"  HORIZONTAL_KERNEL_LENGTH_RATIO: {HORIZONTAL_KERNEL_LENGTH_RATIO}")
    _debug_print(f"  MIN_LINE_WIDTH_RATIO: {MIN_LINE_WIDTH_RATIO}")
    _debug_print(f"  MAX_LINE_HEIGHT: {MAX_LINE_HEIGHT}")
    _debug_print(f"  EROSION_ITERATIONS: {EROSION_ITERATIONS}")
    _debug_print(f"  DILATION_ITERATIONS: {DILATION_ITERATIONS}")
    _debug_print(f"Output image rendering DPI: {OUTPUT_IMAGE_DPI}")


    if not os.path.exists(pdf_file_path):
        print(f"\nERROR: PDF file not found at '{pdf_file_path}'.")
        print(f"Please update the 'pdf_file_path' variable in the script.")
    else:
        _split_pdf_by_detected_lines(pdf_file_path, output_slices_dir)

    print(f"Exam {exam_filename} successfully split into problems")

def _get_exam_pdfs():
    """
    Prompts the user for a folder path, validates it, ensures it contains
    at least one PDF, and then processes each document in the folder
    by calling the `process` function with each filename.
    """
    while True:
        folder_path = input("Please enter the full path to the folder containing your documents: ").strip()

        # Check if the user provided any input
        if not folder_path:
            print("Error: No folder path provided. Please try again.")
            continue

        # 1. Check if the path exists
        if not os.path.exists(folder_path):
            print(f"Error: The path '{folder_path}' does not exist. Please try again.")
            continue

        # 2. Check if the path is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: The path '{folder_path}' is not a folder. Please try again.")
            continue

        # 3. Try to list files and check for at least one PDF
        try:
            all_files_in_folder = os.listdir(folder_path)
        except PermissionError:
            print(f"Error: Permission denied to access the folder '{folder_path}'. Please check your permissions and try again.")
            continue
        except OSError as e:
            # Catch other potential OS errors during listdir
            print(f"Error accessing the folder contents: {e}. Please try again.")
            continue

        if not all_files_in_folder:
            print(f"Error: The folder '{folder_path}' is empty. It must contain at least one PDF file to proceed. Please try again.")
            continue

        # Check for at least one PDF file (case-insensitive)
        pdf_files_found = [f for f in all_files_in_folder if f.lower().endswith(".pdf")]

        if not pdf_files_found:
            print(f"Error: No PDF files found in the folder '{folder_path}'. Please ensure there is at least one PDF and try again.")
            continue

        # If all checks pass:
        print(f"\nSuccessfully validated folder: '{folder_path}'")
        print(f"Found {len(pdf_files_found)} PDF file(s). Processing all {len(all_files_in_folder)} documents in this folder...\n")

        return folder_path, pdf_files_found
    

# --- End helper functions --- #

# --- Public functions --- #
def process_all_exams():
    folder_path, pdf_files_found = _get_exam_pdfs()
    processed_count = 0
    for filename in pdf_files_found:
        full_file_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_file_path): # Ensure it's a file, not a sub-directory
            # print(f"Sending to process(): {filename}") # For debugging
            _process_exam(full_file_path)
            processed_count += 1
        # else:
            # print(f"Skipping directory: {filename}") # Optional: if you want to acknowledge subdirectories

    print(f"\nFinished processing {processed_count} files in '{folder_path}'.")

def get_images_folder() -> str:
    """Returns the name of the folder containing the images of the exam problems"""
    return IMAGES_FOLDER

def delete_problem_images():
    """Deletes all the images in the IMAGES_FOLDER directory.
    Uses shutil.rmtree for recursive deletion."""

    if not IMAGES_FOLDER or not isinstance(IMAGES_FOLDER, str):
        print(f"Error: Invalid folder path provided: {IMAGES_FOLDER}")
        return

    if os.path.exists(IMAGES_FOLDER):
        try:
            if os.path.isdir(IMAGES_FOLDER): # Important: ensure it's a directory
                shutil.rmtree(IMAGES_FOLDER)
                _debug_print(f"Folder '{IMAGES_FOLDER}' and all its contents have been deleted.")
            else:
                print(f"Error: Path '{IMAGES_FOLDER}' is a file, not a folder. Cannot delete as folder.")
        except PermissionError:
            print(f"Error: Permission denied to delete '{IMAGES_FOLDER}'. Check permissions or if files are in use.")
        except Exception as e:
            print(f"An error occurred while deleting '{IMAGES_FOLDER}': {e}")
    else:
        print(f"Info: Folder '{IMAGES_FOLDER}' does not exist. Nothing to delete.")

# --- End public functions --- #
if __name__ == "__main__":
    process_all_exams()