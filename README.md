# Adobe Hackathon Task 1A: PDF Outline Extraction

A high-performance, CPU-optimized solution for extracting structured outlines from PDF documents.

## Overview

This solution extracts document titles and hierarchical headings (H1, H2, H3) from PDF files using a multi-method approach that balances speed and accuracy.

## Features

- **Multi-Method Extraction**: Uses PyMuPDF for existing bookmarks, falls back to font-based analysis
- **Fast Performance**: Optimized for ≤10 second processing of 50-page PDFs
- **CPU Optimized**: Runs efficiently on AMD64 architecture without GPU requirements
- **Robust Analysis**: Intelligent font size and formatting analysis for heading detection
- **Docker Ready**: Containerized solution with all dependencies included

## Architecture

### Method 1: PyMuPDF Outline Extraction (Primary)
- Extracts existing PDF bookmarks/table of contents
- Fastest method when bookmarks are available
- Near-instantaneous processing

### Method 2: Font-Based Analysis (Fallback)
- Analyzes font sizes, styles, and positioning across the document
- Uses statistical analysis to identify heading fonts vs body text
- Groups characters by lines and applies heading detection heuristics

### Method 3: Title Extraction
- Extracts titles from PDF metadata
- Falls back to first-page analysis for largest/centered text
- Smart cleaning and validation of extracted titles

## Algorithm Details

### Font Analysis Heuristics
1. **Statistical Font Analysis**: Identifies most common font size as body text
2. **Heading Font Detection**: Finds fonts significantly larger than body text
3. **Hierarchical Classification**: Maps font sizes to H1, H2, H3 levels
4. **Position-Based Filtering**: Considers text positioning and formatting

### Heading Level Determination
- **H1**: Largest font size, typically centered or prominent
- **H2**: Second largest font size, often bold
- **H3**: Third largest font size, smaller than H2 but larger than body

### Text Cleaning
- Removes page numbers, chapter prefixes, and formatting artifacts
- Validates heading length (3-200 characters)
- Normalizes whitespace and line breaks

## Dependencies

- **PyMuPDF (1.23.27)**: Fast PDF processing and outline extraction
- **pdfplumber (0.11.2)**: Detailed character-level PDF analysis
- **pdfminer.six (20231228)**: Low-level PDF parsing capabilities

## Docker Usage

### Build the Image
```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### Run the Container
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor:latest
```
See [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md) for a full quick start guide.

## Input/Output Format

### Input
- Directory: `/app/input`
- Format: PDF files (*.pdf)
- Limit: Up to 50 pages per PDF

### Output
- Directory: `/app/output`
- Format: JSON files with same name as input PDF
- Structure:
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction", 
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 2
    }
  ]
}
```

## Performance Specifications

- **Processing Time**: ≤10 seconds for 50-page PDF
- **Model Size**: Lightweight libraries, no ML models required
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU**: Utilizes multi-core processing on 8-CPU systems
- **Architecture**: AMD64 (x86_64) compatible

## Error Handling

- Graceful fallback between extraction methods
- Robust error logging and recovery
- Empty results for failed extractions rather than crashes
- Comprehensive input validation

## Optimization Features

- **Lazy Loading**: Processes pages only when needed
- **Memory Efficient**: Closes resources promptly
- **Parallel Ready**: Can be extended for multi-file parallel processing
- **Caching**: Reuses font analysis across pages

## Testing

The solution has been designed to handle various PDF types:
- Academic papers with clear heading hierarchies
- Business documents with embedded bookmarks
- Scanned documents (where text is selectable)
- Multi-language documents
- Complex layouts with mixed formatting

## Limitations

- Requires selectable text (not pure image-based PDFs)
- Heading detection accuracy depends on consistent font usage
- Limited to H1, H2, H3 levels as per requirements
- No network access for enhanced processing

## Future Enhancements

- Multi-language OCR support
- Advanced layout analysis with computer vision
- Machine learning-based heading classification
- Support for tables and figures in outline

---

**Authors**:   
**Created**: 2025  
**License**: MIT (for hackathon use)
