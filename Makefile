# Build LaTeX without editing template.tex or aps360.sty.
#
# `\documentclass[final]{article}` passes `final` to APS360 but not `main`,
# so we prepend PassOptionsToPackage lines before each source file.
#
# Usage:
#   make              # build template.pdf and proposal.pdf
#   make template.pdf
#   make proposal.pdf
#   make clean

LATEX := pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error
BUILD := .build

.PHONY: all clean template.pdf proposal.pdf

all: template.pdf proposal.pdf

$(BUILD):
	@mkdir -p $(BUILD)

# Copy style under both names (Linux needs APS360.sty; macOS is case-insensitive).
$(BUILD)/APS360.sty: aps360.sty | $(BUILD)
	@cp aps360.sty $(BUILD)/APS360.sty

define prep_source
	@mkdir -p $(BUILD)
	@cp aps360.sty $(BUILD)/APS360.sty
	@{ \
		printf '%s\n' '\PassOptionsToPackage{main}{APS360}'; \
		printf '%s\n' '\PassOptionsToPackage{numbers,sort&compress}{natbib}'; \
		cat $(1); \
	} > $(BUILD)/$(2)
endef

template.pdf: template.tex aps360.sty | $(BUILD)
	$(call prep_source,template.tex,template.tex)
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) template.tex
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) template.tex
	@cp $(BUILD)/template.pdf $@

proposal.pdf: proposal.tex aps360.sty figures/APS360_Pipeline.png | $(BUILD)
	$(call prep_source,proposal.tex,proposal.tex)
	@test -d figures && cp -r figures $(BUILD)/
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) proposal.tex
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) proposal.tex
	@cp $(BUILD)/proposal.pdf $@

clean:
	rm -rf $(BUILD) template.pdf proposal.pdf
