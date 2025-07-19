# Adobe Hackathon Task 1A - Build Instructions

## Quick Start

1. **Build the Docker image:**
   ```bash
   docker build --platform linux/amd64 -t pdf-outline-extractor .
   ```

2. **Prepare input directory:**
   ```bash
   mkdir -p input output
   # Copy your PDF files to the input directory
   cp your_pdfs/*.pdf input/
   ```

3. **Run the extraction:**
   ```bash
   docker run --rm \
     -v $(pwd)/input:/app/input \
     -v $(pwd)/output:/app/output \
     --network none \
     pdf-outline-extractor
   ```

4. **Check results:**
   ```bash
   ls output/  # See generated JSON files
   cat output/your_pdf.json  # View extracted outline
   ```


## File Structure
```
project/
├── extract_outline.py    # Main extraction logic
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
├── README.md           # Comprehensive documentation
├── test_extraction.py  # Local testing script
└── .dockerignore       # Build optimization
```
