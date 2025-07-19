#!/usr/bin/env python3
"""
Test script for local development and debugging
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_outline import PDFOutlineExtractor
import json

def test_extraction(pdf_path: str):
    """Test the extraction on a single PDF"""
    extractor = PDFOutlineExtractor()
    result = extractor.extract_outline(pdf_path)

    print(f"=== Results for {Path(pdf_path).name} ===")
    print(f"Title: {result['title']}")
    print(f"Outline items: {len(result['outline'])}")

    for item in result['outline']:
        print(f"  {item['level']}: {item['text']} (page {item['page']})")

    print(f"\nFull JSON output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_extraction.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        print(f"Error: File {pdf_file} does not exist")
        sys.exit(1)

    test_extraction(pdf_file)
