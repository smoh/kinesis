LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out brf
CHECK_RERUN =

NAME = kinesis

all: ${NAME}.pdf


${NAME}.pdf: ${NAME}.tex $(wildcard figures/*.pdf)
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."


clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
	${RM} *Notes.bib

# When actually submitting, the AAS webform does not allow
# nested directories. The following is an example rule to
# collect all files tracked by git in this and figures/ directory
# into one flat directory under the name of choosing.
submit: ${NAME}.pdf
	@read -p "Enter directory name to contain all files:" submitdir; \  
	if [ -d "$$submitdir" ]; then \
		@echo "$$submitdir already exists! Do nothing"; \
		exit 1; \
	fi; \
	mkdir -p $$submitdir \
	cp $$(git ls-tree --name-only HEAD | grep -v ".gitignore\|Makefile\|figures") $$submitdir/ \
	cp $$(git ls-files figures/) $$submitdir/

