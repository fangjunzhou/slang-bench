# Makefile for LaTeX document compilation

# Default target
.PHONY: all clean watch help

# Main document name (change this if your main file has a different name)
MAIN = main

all: $(MAIN).pdf

# Compile the PDF
$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	@if grep -q "\\bibliography" $(MAIN).tex; then \
		biber $(MAIN); \
		pdflatex $(MAIN).tex; \
	fi
	pdflatex $(MAIN).tex

# Watch for changes and recompile automatically
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.nav *.snm *.vrb
	rm -f *.bcf *.run.xml

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Show available targets
help:
	@echo "Available targets:"
	@echo "  all      - Compile the PDF (default)"
	@echo "  watch    - Watch for changes and recompile automatically"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove auxiliary files and PDF"
	@echo "  help     - Show this help message"