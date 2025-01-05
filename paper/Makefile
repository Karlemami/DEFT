MAIN = paper

LATEX = pdflatex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	@echo "Compiling paper.tex...."
	@$(LATEX) $(MAIN).tex
	@echo "Paper compiled!"

open: $(MAIN).pdf
	@open $(MAIN).pdf

clean:
	@echo "Cleaning aux files..."
	@rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg
	@echo "All files cleaned!"

fclean: clean
	@echo "Removing pdf file..."
	rm -f $(MAIN).pdf

.PHONY: all clean fclean
