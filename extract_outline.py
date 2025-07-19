#!/usr/bin/env python3
"""
Adobe Hackathon Task 1A: PDF Outline Extraction
High-performance CPU-based PDF heading extraction using multiple methods
"""

import os
import json
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

# Core PDF processing libraries
import pymupdf  # PyMuPDF for fast outline extraction
import pdfplumber  # For detailed font analysis
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    """
    Multi-method PDF outline extractor optimized for speed and accuracy
    """

    def __init__(self):
        self.font_size_threshold = 2.0  # Minimum font size difference for heading detection
        self.min_heading_chars = 3      # Minimum characters for a valid heading
        self.max_heading_chars = 200    # Maximum characters for a valid heading

    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract document outline using multiple methods
        """
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = time.time()

        try:
            # Method 1: Try PyMuPDF outline extraction (fastest)
            outline_data = self._extract_pymupdf_outline(pdf_path)

            # Method 2: If no outline found, use font-based analysis
            if not outline_data["outline"]:
                logger.info("No embedded outline found, using font-based analysis")
                outline_data = self._extract_font_based_outline(pdf_path)

            # Method 3: Extract title if not found
            if not outline_data["title"]:
                outline_data["title"] = self._extract_title(pdf_path)

            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")

            return outline_data

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}

    def _extract_pymupdf_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract outline using PyMuPDF (fastest method for PDFs with bookmarks)
        """
        try:
            doc = pymupdf.open(pdf_path)
            title = doc.metadata.get('title', '') or ''

            # Get table of contents
            toc = doc.get_toc()  # Returns [[level, title, page], ...]
            outline = []

            for level, heading_title, page in toc:
                # Convert level to H1, H2, H3 format
                if level <= 3:
                    level_str = f"H{level}"
                    clean_title = self._clean_heading_text(heading_title)
                    if clean_title:
                        outline.append({
                            "level": level_str,
                            "text": clean_title,
                            "page": page
                        })

            doc.close()
            logger.info(f"PyMuPDF found {len(outline)} outline items")

            return {
                "title": self._clean_title_text(title),
                "outline": outline
            }

        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            return {"title": "", "outline": []}

    def _extract_font_based_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract outline using font-based analysis with pdfplumber
        """
        try:
            outline = []
            title = ""

            with pdfplumber.open(pdf_path) as pdf:
                # Collect font information across all pages
                font_stats = self._analyze_font_statistics(pdf)
                heading_fonts = self._identify_heading_fonts(font_stats)

                logger.info(f"Identified {len(heading_fonts)} heading font sizes: {list(heading_fonts.keys())}")

                for page_num, page in enumerate(pdf.pages, 1):
                    page_headings = self._extract_page_headings(page, heading_fonts, page_num)
                    outline.extend(page_headings)

                # Extract title from first page if not found
                if not title and pdf.pages:
                    title = self._extract_title_from_page(pdf.pages[0])

            # Sort outline by page and position
            outline.sort(key=lambda x: (x["page"], x.get("y_position", 0)))

            logger.info(f"Font-based analysis found {len(outline)} headings")

            return {
                "title": title,
                "outline": outline
            }

        except Exception as e:
            logger.error(f"Font-based extraction failed: {str(e)}")
            return {"title": "", "outline": []}

    def _analyze_font_statistics(self, pdf) -> Dict[float, Dict]:
        """
        Analyze font usage statistics across the document
        """
        font_stats = defaultdict(lambda: {
            'count': 0, 
            'avg_chars_per_line': 0, 
            'is_bold': False, 
            'sample_text': '',
            'y_positions': []
        })

        total_chars = 0

        for page in pdf.pages:
            for char in page.chars:
                font_size = round(char['size'], 1)
                font_name = char.get('fontname', '')

                font_stats[font_size]['count'] += 1
                font_stats[font_size]['is_bold'] = font_stats[font_size]['is_bold'] or ('bold' in font_name.lower())

                if len(font_stats[font_size]['sample_text']) < 50:
                    font_stats[font_size]['sample_text'] += char['text']

                font_stats[font_size]['y_positions'].append(char['top'])
                total_chars += 1

        # Calculate relative frequency
        for size in font_stats:
            font_stats[size]['frequency'] = font_stats[size]['count'] / total_chars if total_chars > 0 else 0

        return dict(font_stats)

    def _identify_heading_fonts(self, font_stats: Dict[float, Dict]) -> Dict[float, str]:
        """
        Identify which font sizes correspond to headings
        """
        if not font_stats:
            return {}

        # Sort fonts by size (descending)
        sorted_fonts = sorted(font_stats.keys(), reverse=True)

        # Find the most common font size (likely body text)
        body_font_size = max(font_stats.keys(), key=lambda x: font_stats[x]['frequency'])

        heading_fonts = {}
        level = 1

        for font_size in sorted_fonts:
            stats = font_stats[font_size]

            # Skip if too similar to body text size or too small frequency
            if (abs(font_size - body_font_size) < self.font_size_threshold and 
                stats['frequency'] > 0.3):
                continue

            # Consider as heading if:
            # 1. Larger than body text
            # 2. Has reasonable frequency (not too rare, not too common)
            # 3. Has some bold formatting
            if (font_size > body_font_size + self.font_size_threshold and
                0.001 < stats['frequency'] < 0.1 and
                level <= 3):

                heading_fonts[font_size] = f"H{level}"
                level += 1

        return heading_fonts

    def _extract_page_headings(self, page, heading_fonts: Dict[float, str], page_num: int) -> List[Dict]:
        """
        Extract headings from a single page
        """
        headings = []

        # Group characters by line
        lines = self._group_chars_by_line(page.chars)

        for line in lines:
            line_text = ''.join(char['text'] for char in line).strip()

            if not line_text or len(line_text) < self.min_heading_chars:
                continue

            # Check if this line uses a heading font
            line_font_sizes = [round(char['size'], 1) for char in line if char['size']]

            if not line_font_sizes:
                continue

            primary_font_size = Counter(line_font_sizes).most_common(1)[0][0]

            if primary_font_size in heading_fonts:
                clean_text = self._clean_heading_text(line_text)

                if clean_text:
                    headings.append({
                        "level": heading_fonts[primary_font_size],
                        "text": clean_text,
                        "page": page_num,
                        "y_position": min(char['top'] for char in line),
                        "font_size": primary_font_size
                    })

        return headings

    def _group_chars_by_line(self, chars) -> List[List[Dict]]:
        """
        Group characters by line based on y-position
        """
        if not chars:
            return []

        # Sort by y-position then x-position
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))

        lines = []
        current_line = [sorted_chars[0]]
        current_y = sorted_chars[0]['top']

        for char in sorted_chars[1:]:
            # If y-position is close to current line, add to current line
            if abs(char['top'] - current_y) < 5:  # 5 point tolerance
                current_line.append(char)
            else:
                # Start new line
                lines.append(current_line)
                current_line = [char]
                current_y = char['top']

        if current_line:
            lines.append(current_line)

        return lines

    def _extract_title(self, pdf_path: str) -> str:
        """
        Extract document title using multiple methods
        """
        try:
            # Method 1: PDF metadata
            doc = pymupdf.open(pdf_path)
            title = doc.metadata.get('title', '') or ''
            doc.close()

            if title:
                return self._clean_title_text(title)

            # Method 2: First page analysis
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.pages:
                    title = self._extract_title_from_page(pdf.pages[0])

            return title or ""

        except Exception as e:
            logger.warning(f"Title extraction failed: {str(e)}")
            return ""

    def _extract_title_from_page(self, page) -> str:
        """
        Extract title from first page by finding largest/centered text
        """
        try:
            lines = self._group_chars_by_line(page.chars)

            if not lines:
                return ""

            # Look for large, centered text in upper portion of page
            candidates = []
            page_height = page.height
            page_width = page.width

            for line in lines[:10]:  # Only check first 10 lines
                line_text = ''.join(char['text'] for char in line).strip()

                if (len(line_text) < self.min_heading_chars or 
                    len(line_text) > self.max_heading_chars):
                    continue

                # Calculate line metrics
                avg_font_size = sum(char['size'] for char in line) / len(line)
                line_y = min(char['top'] for char in line)
                line_x_center = (min(char['x0'] for char in line) + max(char['x1'] for char in line)) / 2

                # Score based on: font size, position (higher = better), centering
                size_score = avg_font_size / 12  # Normalize to typical body size
                position_score = (page_height - line_y) / page_height  # Higher on page = better
                center_score = 1 - abs(line_x_center - page_width/2) / (page_width/2)  # More centered = better

                total_score = size_score * 0.4 + position_score * 0.3 + center_score * 0.3

                candidates.append((line_text, total_score))

            if candidates:
                # Return the highest scoring candidate
                best_title = max(candidates, key=lambda x: x[1])[0]
                return self._clean_title_text(best_title)

        except Exception as e:
            logger.warning(f"Page title extraction failed: {str(e)}")

        return ""

    def _clean_heading_text(self, text: str) -> str:
        """
        Clean and validate heading text
        """
        if not text:
            return ""

        # Remove extra whitespace and line breaks
        cleaned = ' '.join(text.split())

        # Remove common prefixes/suffixes (using raw strings to avoid warnings)
        cleaned = re.sub(r'^(Chapter|Section|Part)\s*\d*\.?\s*', '', cleaned, flags=re.IGNORECASE)

        # Remove page numbers at the end
        cleaned = re.sub(r'\s+\d+\s*$', '', cleaned)

        # Keep only if within reasonable length
        if self.min_heading_chars <= len(cleaned) <= self.max_heading_chars:
            return cleaned

        return ""

    def _clean_title_text(self, title: str) -> str:
        """
        Clean and validate title text
        """
        if not title:
            return ""

        # Remove extra whitespace
        cleaned = ' '.join(title.split())

        # Remove file extensions (using raw string)
        cleaned = re.sub(r'\.(pdf|doc|docx)$', '', cleaned, flags=re.IGNORECASE)

        return cleaned[:200] if cleaned else ""  # Limit title length

def process_pdfs():
    """
    Main processing function - process all PDFs in input directory
    """
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return

    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    extractor = PDFOutlineExtractor()

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")

            # Extract outline
            result = extractor.extract_outline(str(pdf_file))

            # Save result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved result to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            # Save empty result for failed files
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"title": "", "outline": []}, f, indent=2)

if __name__ == "__main__":
    logger.info("Starting PDF Outline Extraction")
    process_pdfs()
    logger.info("Processing complete")
