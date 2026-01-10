"""
Visual evaluator for Design2Code benchmark.

Adapted from the original visual_score.py to work with HTML strings and PIL Images
instead of file paths.
"""
import io
import logging
import re
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from bs4 import BeautifulSoup, Comment, NavigableString
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from difflib import SequenceMatcher
from PIL import Image
from playwright.async_api import async_playwright
from scipy.optimize import linear_sum_assignment
from collections import Counter
from copy import deepcopy

logger = logging.getLogger(__name__)

# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)

# CLIP model loading - lazy initialization
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        try:
            import torch
            logger.debug("Torch imported successfully")
            try:
                import clip
                logger.debug("CLIP library imported successfully")
            except ImportError as e:
                logger.error(f"CLIP library import failed: {e}")
                logger.warning("CLIP library not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
                _clip_model = None
                return None, None, None

            _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Loading CLIP model 'ViT-B/32' on {_clip_device}...")

            # Try to load CLIP model, handling SSL certificate issues
            try:
                _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
            except (urllib.error.URLError, ssl.SSLError) as ssl_error:
                if "CERTIFICATE_VERIFY_FAILED" in str(ssl_error) or isinstance(ssl_error, ssl.SSLError):
                    logger.warning(f"SSL certificate verification failed. Retrying with SSL verification disabled...")

                    # Create unverified SSL context
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    # Monkey-patch urllib to use unverified context temporarily
                    original_urlopen = urllib.request.urlopen
                    def urlopen_with_ssl_bypass(*args, **kwargs):
                        if 'context' not in kwargs:
                            kwargs['context'] = ssl_context
                        return original_urlopen(*args, **kwargs)

                    urllib.request.urlopen = urlopen_with_ssl_bypass
                    try:
                        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
                        logger.debug("CLIP model loaded successfully with SSL verification disabled")
                    finally:
                        # Restore original urlopen
                        urllib.request.urlopen = original_urlopen
                else:
                    raise

            _clip_model.eval()  # Set to evaluation mode
            logger.debug(f"CLIP model loaded successfully on {_clip_device}")
        except Exception as e:
            logger.error(f"CLIP model loading failed: {e}", exc_info=True)
            logger.warning(f"CLIP not available: {e}. CLIP similarity will return 0.0")
            _clip_model = None
    else:
        logger.debug(f"CLIP model already loaded on {_clip_device}")
    return _clip_model, _clip_preprocess, _clip_device


async def html_to_screenshot(html_content: str, width: int = 1920, height: int = 1080) -> Image.Image:
    """
    Convert HTML content to a screenshot using Playwright.

    Args:
        html_content: HTML string to render
        width: Viewport width
        height: Viewport height

    Returns:
        PIL Image of the screenshot
    """
    import shutil

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})

        # Create a temporary directory to hold both HTML and assets
        temp_dir = tempfile.mkdtemp()
        html_path = Path(temp_dir) / "index.html"
        rick_jpg_path = Path(temp_dir) / "rick.jpg"

        # Find rick.jpg in assets folder
        # Path structure: scenarios/design2code/evaluation/visual_evaluator.py
        # Workdir is typically at project root (tutorial/), so assets is at root/assets/
        project_root = Path(__file__).parent.parent.parent.parent
        assets_rick = project_root / "assets" / "rick.jpg"

        # If not found, try alternative path structure (if running from different location)
        if not assets_rick.exists():
            # Try from current working directory
            cwd_assets = Path.cwd() / "assets" / "rick.jpg"
            if cwd_assets.exists():
                assets_rick = cwd_assets

        try:
            # Write HTML to temporary file
            html_path.write_text(html_content, encoding='utf-8')

            # Copy rick.jpg to temp directory if it exists, so HTML can access it
            if assets_rick.exists():
                shutil.copy2(assets_rick, rick_jpg_path)
                logger.debug(f"Copied rick.jpg from {assets_rick} to {rick_jpg_path}")
            else:
                logger.warning(f"rick.jpg not found at {assets_rick}, images may not render correctly")

            # Load the HTML file
            await page.goto(f"file://{html_path}", timeout=60000)
            await page.wait_for_load_state("networkidle", timeout=5000)

            # Take screenshot with full page and animations disabled
            screenshot_bytes = await page.screenshot(full_page=True, animations="disabled", timeout=60000)
            await browser.close()

            # Convert to PIL Image
            img = Image.open(io.BytesIO(screenshot_bytes))
            return img.convert('RGB')
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_color_from_style(style: str) -> tuple[int, int, int]:
    """Extract RGB color from CSS style string."""
    # Try rgb() format
    color_match = re.search(r'color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
    if color_match:
        return tuple(int(color_match.group(i)) for i in range(1, 4))

    # Try hex format #RRGGBB
    color_match = re.search(r'color:\s*#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})', style)
    if color_match:
        return tuple(int(color_match.group(i), 16) for i in range(1, 4))

    # Try short hex #RGB
    color_match = re.search(r'color:\s*#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])\b', style)
    if color_match:
        return tuple(int(color_match.group(i) * 2, 16) for i in range(1, 4))

    # Try named colors (basic set)
    named_colors = {
        'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
        'green': (0, 128, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'gray': (128, 128, 128),
        'grey': (128, 128, 128), 'orange': (255, 165, 0), 'purple': (128, 0, 128)
    }
    for name, rgb in named_colors.items():
        if re.search(rf'color:\s*{name}\b', style, re.IGNORECASE):
            return rgb

    return (0, 0, 0)  # Default black


def extract_blocks_from_html(html_content: str, image_width: int = 1920, image_height: int = 1080) -> list[dict]:
    """
    Extract text blocks from HTML content by parsing the DOM.

    This extracts text elements with their approximate positions and colors.
    Uses a hierarchical approach to estimate positions based on DOM structure.

    Args:
        html_content: HTML string to parse
        image_width: Width of the rendered image (for normalized coordinates)
        image_height: Height of the rendered image (for normalized coordinates)

    Returns:
        List of block dictionaries with 'bbox', 'text', 'color' keys
    """
    blocks = []
    if not html_content or not html_content.strip():
        logger.debug("extract_blocks_from_html: Empty HTML content")
        return blocks

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.warning(f"extract_blocks_from_html: Failed to parse HTML: {e}")
        return blocks

    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link", "title"]):
        script.decompose()

    # Track vertical position as we traverse
    y_position = 0.05  # Start a bit from top
    block_counter = 0

    def get_text_from_element(element) -> str:
        """Extract all text from an element, excluding nested block elements."""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            return text if text else ""

        text_parts = []
        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    text_parts.append(text)
            elif hasattr(child, 'name'):
                # For inline elements, get text directly
                if child.name in ['span', 'a', 'strong', 'em', 'b', 'i', 'u', 'code', 'small', 'label', 'button']:
                    child_text = child.get_text(separator=' ', strip=True)
                    if child_text:
                        text_parts.append(child_text)
                # For block elements, we'll process them separately
        result = ' '.join(text_parts).strip()
        return result

    def extract_color_from_element(element) -> tuple[int, int, int]:
        """Extract color from element's style or parent's style."""
        if hasattr(element, 'get'):
            style = element.get('style', '')
            if style:
                color = extract_color_from_style(style)
                if color != (0, 0, 0) or 'color' in style.lower():
                    return color

        # Check parent
        if hasattr(element, 'parent') and element.parent:
            return extract_color_from_element(element.parent)

        return (0, 0, 0)  # Default black

    # Find all text-containing block-level elements
    block_elements = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section',
                      'article', 'header', 'footer', 'nav', 'main', 'aside', 'li', 'td', 'th',
                      'label', 'button', 'span']  # Added span and label as they can contain significant text

    for element in soup.find_all(block_elements):
        text = get_text_from_element(element)
        if text and len(text) > 0:
            # Estimate bounding box
            # Width: assume most content uses 60-80% of screen width
            width = min(0.8, max(0.3, len(text) * 0.008))
            # Height: estimate based on text length and element type
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                height = 0.05
            else:
                # Rough estimate: ~50 chars per line, ~0.03 height per line
                lines = max(1, len(text) / 50)
                height = min(0.2, max(0.03, lines * 0.03))

            # X position: center or left-align (simplified)
            x = 0.1

            # Y position: stack vertically
            y = min(0.95, y_position)
            y_position += height + 0.02  # Add spacing

            bbox = (x, y, width, height)
            color = extract_color_from_element(element)

            blocks.append({
                'text': text,
                'bbox': bbox,
                'color': color
            })
            block_counter += 1

    # If no block elements found, extract from any text nodes (fallback)
    if len(blocks) == 0:
        # Try to get all text from body or html
        body = soup.find('body') or soup.find('html') or soup
        if body:
            all_text = body.get_text(separator=' ', strip=True)
            if all_text:
                # Split by common separators (periods, newlines, etc.)
                # First try splitting by newlines
                lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                if len(lines) == 0:
                    # If no newlines, try splitting by periods
                    lines = [line.strip() + '.' for line in all_text.split('.') if line.strip()]
                if len(lines) == 0:
                    # If still nothing, use the whole text as one block
                    lines = [all_text] if all_text else []

                for i, line in enumerate(lines[:50]):  # Limit to 50 lines
                    if line and len(line.strip()) > 0:
                        blocks.append({
                            'text': line.strip(),
                            'bbox': (0.1, 0.05 + i * 0.04, min(0.8, max(0.3, len(line) * 0.008)), 0.035),
                            'color': (0, 0, 0)
                        })

        # If still no blocks, try getting any direct text nodes
        if len(blocks) == 0:
            for text_node in soup.find_all(string=True):
                text = str(text_node).strip()
                if text and len(text) > 2:  # Ignore very short text nodes
                    blocks.append({
                        'text': text,
                        'bbox': (0.1, 0.05 + len(blocks) * 0.04, min(0.8, max(0.3, len(text) * 0.008)), 0.035),
                        'color': (0, 0, 0)
                    })
                    if len(blocks) >= 50:  # Limit
                        break

    if len(blocks) == 0:
        logger.warning(f"extract_blocks_from_html: No blocks extracted. HTML length: {len(html_content)}, HTML preview: {html_content[:200]}")

    return blocks


def get_blocks_ocr_free_simple(image: Image.Image, html_content: Optional[str] = None) -> list[dict]:
    """
    Extract blocks from image or HTML.

    If html_content is provided, extracts blocks from HTML (more reliable).
    Otherwise, attempts basic image-based extraction (limited).

    Args:
        image: PIL Image (used for dimensions if html_content provided)
        html_content: Optional HTML string to extract blocks from

    Returns:
        List of block dictionaries with 'bbox', 'text', 'color' keys
    """
    # If HTML content is available, use HTML-based extraction (more reliable)
    if html_content:
        width, height = image.size
        return extract_blocks_from_html(html_content, width, height)

    # Fallback: basic image-based extraction would go here
    # For now, return empty list if no HTML provided
    logger.warning("get_blocks_ocr_free_simple: No HTML content provided, cannot extract blocks from image alone")
    return []


def check_repetitive_content(html_file: str) -> None:
    """
    Check and fix repetitive content in HTML file.
    Stub implementation.

    Args:
        html_file: Path to HTML file
    """
    # TODO: Implement repetitive content checking
    pass


def calculate_similarity(block1: dict, block2: dict, max_distance: float = 1.42) -> float:
    """Calculate text similarity between two blocks."""
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    return text_similarity


def adjust_cost_for_context(cost_matrix: np.ndarray, consecutive_bonus: float = 1.0, window_size: int = 20) -> np.ndarray:
    """Adjust cost matrix for context."""
    if window_size <= 0:
        return cost_matrix

    n, m = cost_matrix.shape
    adjusted_cost_matrix = np.copy(cost_matrix)

    for i in range(n):
        for j in range(m):
            if adjusted_cost_matrix[i][j] >= -0.5:
                continue
            nearby_matrix = cost_matrix[
                max(0, i - window_size):min(n, i + window_size + 1),
                max(0, j - window_size):min(m, j + window_size + 1)
            ]
            flattened_array = nearby_matrix.flatten()
            sorted_array = np.sort(flattened_array)[::-1]
            sorted_array = np.delete(sorted_array, np.where(sorted_array == cost_matrix[i, j])[0][0])
            top_k_elements = sorted_array[- window_size * 2:]
            sum_top_k = np.sum(top_k_elements)
            bonus = consecutive_bonus * sum_top_k
            adjusted_cost_matrix[i][j] += bonus
    return adjusted_cost_matrix


def create_cost_matrix(A: list[dict], B: list[dict]) -> np.ndarray:
    """Create cost matrix for matching blocks."""
    n = len(A)
    m = len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = -calculate_similarity(A[i], B[j])
    return cost_matrix


def calculate_distance_max_1d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate max 1D distance."""
    return max(abs(x2 - x1), abs(y2 - y1))


def calculate_ratio(h1: float, h2: float) -> float:
    """Calculate ratio of two heights."""
    return max(h1, h2) / min(h1, h2) if min(h1, h2) > 0 else float('inf')


def rgb_to_lab(rgb: tuple[int, int, int]) -> LabColor:
    """Convert RGB color to Lab color space."""
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color


def color_similarity_ciede2000(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """Calculate color similarity using CIEDE2000 formula."""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    delta_e = delta_e_cie2000(lab1, lab2)
    similarity = max(0, 1 - (delta_e / 100))
    return similarity


def merge_blocks_wo_check(block1: dict, block2: dict) -> dict:
    """Merge two blocks without checking."""
    merged_text = block1['text'] + " " + block2['text']

    x_min = min(block1['bbox'][0], block2['bbox'][0])
    y_min = min(block1['bbox'][1], block2['bbox'][1])
    x_max = max(block1['bbox'][0] + block1['bbox'][2], block2['bbox'][0] + block2['bbox'][2])
    y_max = max(block1['bbox'][1] + block1['bbox'][3], block2['bbox'][1] + block2['bbox'][3])
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    merged_color = tuple(
        (color1 + color2) // 2 for color1, color2 in zip(block1['color'], block2['color'])
    )

    return {'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}


def find_maximum_matching(A: list[dict], B: list[dict], consecutive_bonus: float, window_size: int) -> tuple:
    """Find maximum matching between blocks."""
    cost_matrix = create_cost_matrix(A, B)
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_cost = cost_matrix[row_ind, col_ind].tolist()
    return list(zip(row_ind, col_ind)), current_cost, cost_matrix


def remove_indices(lst: list, indices: list[int]) -> list:
    """Remove indices from list."""
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst


def merge_blocks_by_list(blocks: list[dict], merge_list: list[list[int]]) -> list[dict]:
    """Merge blocks according to merge list."""
    pop_list = []
    while True:
        if len(merge_list) == 0:
            remove_indices(blocks, pop_list)
            return blocks

        i = merge_list[0][0]
        j = merge_list[0][1]

        blocks[i] = merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)

        merge_list.pop(0)
        if len(merge_list) > 0:
            new_merge_list = []
            for k in range(len(merge_list)):
                if merge_list[k][0] != i and merge_list[k][1] != i and merge_list[k][0] != j and merge_list[k][1] != j:
                    new_merge_list.append(merge_list[k])
            merge_list = new_merge_list


def difference_of_means(list1: list[float], list2: list[float]) -> float:
    """Calculate difference of means after removing common elements."""
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    if mean_list1 - mean_list2 > 0:
        if unique_list1 and unique_list2 and min(unique_list1) > min(unique_list2):
            return mean_list1 - mean_list2
        else:
            return 0.0
    else:
        return mean_list1 - mean_list2


def find_possible_merge(A: list[dict], B: list[dict], consecutive_bonus: float, window_size: int, debug: bool = False) -> tuple:
    """Find possible block merges to optimize matching."""
    merge_bonus = 0.0
    merge_windows = 1

    def sortFn(value):
        return value[2]

    while True:
        A_changed = False
        B_changed = False

        matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)

        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)

                updated_matching, updated_cost, _ = find_maximum_matching(new_A, B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)

        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)

                updated_matching, updated_cost, _ = find_maximum_matching(A, new_B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)

        if not A_changed and not B_changed:
            break

    matching, _, _ = find_maximum_matching(A, B, consecutive_bonus, window_size)
    return A, B, matching


def merge_blocks_by_bbox(blocks: list[dict]) -> list[dict]:
    """Merge blocks with the same bounding box."""
    merged_blocks = {}

    for block in blocks:
        bbox = tuple(block['bbox'])
        if bbox in merged_blocks:
            existing_block = merged_blocks[bbox]
            existing_block['text'] += ' ' + block['text']
            existing_block['color'] = [(ec + c) / 2 for ec, c in zip(existing_block['color'], block['color'])]
        else:
            merged_blocks[bbox] = block

    return list(merged_blocks.values())


def mask_bounding_boxes_with_inpainting(image: Image.Image, bounding_boxes: list[tuple]) -> Image.Image:
    """Mask bounding boxes with inpainting."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    height, width = image_cv.shape[:2]

    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        w = int(w_ratio * width)
        h = int(h_ratio * height)
        mask[y:y+h, x:x+w] = 255

    inpainted_image = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    return inpainted_image_pil


def rescale_and_mask(image: Image.Image, blocks: list[dict]) -> Image.Image:
    """Rescale and mask image."""
    if len(blocks) > 0:
        image = mask_bounding_boxes_with_inpainting(image, [block['bbox'] for block in blocks])

    width, height = image.size

    if width < height:
        new_size = (width, width)
    else:
        new_size = (height, height)

    img_resized = image.resize(new_size, Image.LANCZOS)
    return img_resized


def calculate_clip_similarity_with_blocks(image1: Image.Image, image2: Image.Image, blocks1: list[dict], blocks2: list[dict]) -> float:
    """Calculate CLIP similarity with blocks."""
    model, preprocess, device = _get_clip_model()
    if model is None:
        logger.warning("CLIP model is None, returning 0.0 for similarity")
        return 0.0

    try:
        import torch

        logger.debug(f"Calculating CLIP similarity: image1 size={image1.size}, image2 size={image2.size}, blocks1={len(blocks1)}, blocks2={len(blocks2)}")

        image1_processed = rescale_and_mask(image1, blocks1)
        image2_processed = rescale_and_mask(image2, blocks2)

        logger.debug(f"Processed images: image1 size={image1_processed.size}, image2 size={image2_processed.size}")

        image1_tensor = preprocess(image1_processed).unsqueeze(0).to(device)
        image2_tensor = preprocess(image2_processed).unsqueeze(0).to(device)

        logger.debug(f"Image tensors created: image1 shape={image1_tensor.shape}, image2 shape={image2_tensor.shape}, device={device}")

        with torch.no_grad():
            image_features1 = model.encode_image(image1_tensor)
            image_features2 = model.encode_image(image2_tensor)

        logger.debug(f"Image features extracted: features1 shape={image_features1.shape}, features2 shape={image_features2.shape}")

        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

        similarity = (image_features1 @ image_features2.T).item()
        return similarity
    except Exception as e:
        logger.error(f"CLIP similarity calculation failed: {e}", exc_info=True)
        return 0.0


def truncate_repeated_html_elements(soup: BeautifulSoup, max_count: int = 50) -> str:
    """Truncate repeated HTML elements."""
    content_counts = {}

    for element in soup.find_all(True):
        if isinstance(element, (NavigableString, Comment)):
            continue

        try:
            element_html = str(element)
        except:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1

        if content_counts[element_html] > max_count:
            element.decompose()

    return str(soup)


def pre_process_html(html_content: str) -> str:
    """Pre-process HTML content (string-based version of original pre_process)."""
    # Ensure HTML structure (equivalent to make_html)
    if not re.search(r'<html[^>]*>', html_content, re.IGNORECASE):
        html_content = f'<html><body><p>{html_content}</p></body></html>'

    # Parse and truncate repeated elements (equivalent to original pre_process)
    soup = BeautifulSoup(html_content, 'html.parser')
    soup_str = truncate_repeated_html_elements(soup)

    return soup_str


async def evaluate_html(generated_html: str, reference_html: str, reference_image: Optional[Image.Image] = None) -> dict[str, float]:
    """
    Evaluate generated HTML against reference HTML/image.

    Args:
        generated_html: Generated HTML string
        reference_html: Reference HTML string
        reference_image: Optional reference screenshot (PIL Image)

    Returns:
        Dictionary with detailed scores:
        - 'overall_score': Overall evaluation score (0.0 to 1.0)
        - 'size_score': Size/area matching score
        - 'text_score': Text similarity score
        - 'position_score': Position similarity score
        - 'color_score': Color similarity score
        - 'clip_score': CLIP similarity score
    """
    try:
        # Pre-process HTML
        logger.debug(f"evaluate_html: Original generated_html length: {len(generated_html)}")
        logger.debug(f"evaluate_html: Original reference_html length: {len(reference_html)}")

        generated_html_processed = pre_process_html(generated_html)
        logger.debug(f"evaluate_html: Processed generated_html length: {len(generated_html_processed)}")

        reference_html_processed = pre_process_html(reference_html)
        logger.debug(f"evaluate_html: Processed reference_html length: {len(reference_html_processed)}")

        # Generate screenshots
        generated_image = await html_to_screenshot(generated_html_processed)
        if reference_image is None:
            reference_image = await html_to_screenshot(reference_html_processed)

        # Extract blocks from HTML (more reliable than image-based extraction)
        logger.debug(f"evaluate_html: About to extract blocks from generated_html (length: {len(generated_html_processed)})")
        generated_blocks = get_blocks_ocr_free_simple(generated_image, generated_html_processed)
        logger.debug(f"evaluate_html: About to extract blocks from reference_html (length: {len(reference_html_processed)})")
        reference_blocks = get_blocks_ocr_free_simple(reference_image, reference_html_processed)

        # If no blocks extracted, fall back to CLIP similarity only
        if len(generated_blocks) == 0 or len(reference_blocks) == 0:
            logger.warning("No blocks extracted, using CLIP similarity only")
            clip_score = calculate_clip_similarity_with_blocks(generated_image, reference_image, [], [])
            return {
                'overall_score': float(0.2 * clip_score),
                'size_score': 0.0,
                'text_score': 0.0,
                'position_score': 0.0,
                'color_score': 0.0,
                'clip_score': float(clip_score),
            }

        # Merge blocks by bbox
        generated_blocks = merge_blocks_by_bbox(generated_blocks)
        reference_blocks = merge_blocks_by_bbox(reference_blocks)

        # Find matching
        consecutive_bonus, window_size = 0.1, 1
        gen_blocks_m, ref_blocks_m, matching = find_possible_merge(
            generated_blocks, deepcopy(reference_blocks), consecutive_bonus, window_size
        )

        # Filter matching by similarity
        filtered_matching = []
        for i, j in matching:
            text_similarity = SequenceMatcher(None, gen_blocks_m[i]['text'], ref_blocks_m[j]['text']).ratio()
            if text_similarity < 0.5:
                continue
            filtered_matching.append([i, j, text_similarity])
        matching = filtered_matching

        indices1 = [item[0] for item in matching]
        indices2 = [item[1] for item in matching]

        sum_areas = []
        matched_areas = []
        matched_text_scores = []
        position_scores = []
        text_color_scores = []

        # Calculate unmatched areas
        unmatched_area_1 = 0.0
        for i in range(len(gen_blocks_m)):
            if i not in indices1:
                unmatched_area_1 += gen_blocks_m[i]['bbox'][2] * gen_blocks_m[i]['bbox'][3]
        unmatched_area_2 = 0.0
        for j in range(len(ref_blocks_m)):
            if j not in indices2:
                unmatched_area_2 += ref_blocks_m[j]['bbox'][2] * ref_blocks_m[j]['bbox'][3]
        sum_areas.append(unmatched_area_1 + unmatched_area_2)

        # Calculate scores for matched blocks
        for i, j, text_similarity in matching:
            sum_block_area = gen_blocks_m[i]['bbox'][2] * gen_blocks_m[i]['bbox'][3] + ref_blocks_m[j]['bbox'][2] * ref_blocks_m[j]['bbox'][3]

            # Calculate position similarity (normalized distance, clamped to [0, 1])
            distance = calculate_distance_max_1d(
                gen_blocks_m[i]['bbox'][0] + gen_blocks_m[i]['bbox'][2] / 2,
                gen_blocks_m[i]['bbox'][1] + gen_blocks_m[i]['bbox'][3] / 2,
                ref_blocks_m[j]['bbox'][0] + ref_blocks_m[j]['bbox'][2] / 2,
                ref_blocks_m[j]['bbox'][1] + ref_blocks_m[j]['bbox'][3] / 2
            )
            # Clamp to [0, 1] since coordinates are normalized
            position_similarity = max(0.0, min(1.0, 1.0 - distance))

            text_color_similarity = color_similarity_ciede2000(gen_blocks_m[i]['color'], ref_blocks_m[j]['color'])

            sum_areas.append(sum_block_area)
            matched_areas.append(sum_block_area)
            matched_text_scores.append(text_similarity)
            position_scores.append(position_similarity)
            text_color_scores.append(text_color_similarity)

        if len(matched_areas) > 0:
            sum_sum_areas = np.sum(sum_areas)

            final_size_score = np.sum(matched_areas) / np.sum(sum_areas)
            final_matched_text_score = np.mean(matched_text_scores)
            final_position_score = np.mean(position_scores)
            final_text_color_score = np.mean(text_color_scores)
            final_clip_score = calculate_clip_similarity_with_blocks(generated_image, reference_image, generated_blocks, reference_blocks)
            final_score = 0.2 * (final_size_score + final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)

            return {
                'overall_score': float(final_score),
                'size_score': float(final_size_score),
                'text_score': float(final_matched_text_score),
                'position_score': float(final_position_score),
                'color_score': float(final_text_color_score),
                'clip_score': float(final_clip_score),
            }
        else:
            logger.warning("No matched blocks, using CLIP similarity only")
            final_clip_score = calculate_clip_similarity_with_blocks(generated_image, reference_image, generated_blocks, reference_blocks)
            clip_only_score = 0.2 * final_clip_score
            return {
                'overall_score': float(clip_only_score),
                'size_score': 0.0,
                'text_score': 0.0,
                'position_score': 0.0,
                'color_score': 0.0,
                'clip_score': float(final_clip_score),
            }

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return {
            'overall_score': 0.0,
            'size_score': 0.0,
            'text_score': 0.0,
            'position_score': 0.0,
            'color_score': 0.0,
            'clip_score': 0.0,
        }
