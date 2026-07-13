# Build LaTeX without editing template.tex or aps360.sty.
#
# `\documentclass[final]{article}` passes `final` to APS360 but not `main`,
# so we prepend PassOptionsToPackage lines before each source file.
#
# Usage:
#   make              # build template.pdf, proposal.pdf, and progress_report.pdf
#   make template.pdf
#   make proposal.pdf
#   make progress_report.pdf
#   make clean

LATEX := pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error
BUILD := .build
INSTRUCTIONS := instructions
DELIVERABLE := deliverable

.PHONY: all clean template.pdf proposal.pdf progress_report.pdf

all: template.pdf proposal.pdf progress_report.pdf

$(BUILD):
	@mkdir -p $(BUILD)

# Copy style under both names (Linux needs APS360.sty; macOS is case-insensitive).
$(BUILD)/APS360.sty: $(INSTRUCTIONS)/aps360.sty | $(BUILD)
	@cp $(INSTRUCTIONS)/aps360.sty $(BUILD)/APS360.sty

define prep_source
	@mkdir -p $(BUILD)
	@cp $(INSTRUCTIONS)/aps360.sty $(BUILD)/APS360.sty
	@{ \
		printf '%s\n' '\PassOptionsToPackage{main}{APS360}'; \
		printf '%s\n' '\PassOptionsToPackage{numbers,sort&compress}{natbib}'; \
		cat $(1); \
	} > $(BUILD)/$(2)
endef

template.pdf: $(INSTRUCTIONS)/template.tex $(INSTRUCTIONS)/aps360.sty | $(BUILD)
	$(call prep_source,$(INSTRUCTIONS)/template.tex,template.tex)
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) template.tex
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) template.tex
	@cp $(BUILD)/template.pdf $(INSTRUCTIONS)/$@

proposal.pdf: $(DELIVERABLE)/proposal.tex $(INSTRUCTIONS)/aps360.sty figures/APS360_Pipeline.png | $(BUILD)
	$(call prep_source,$(DELIVERABLE)/proposal.tex,proposal.tex)
	@test -d figures && cp -r figures $(BUILD)/
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) proposal.tex
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) proposal.tex
	@cp $(BUILD)/proposal.pdf $(DELIVERABLE)/$@

progress_report.pdf: $(DELIVERABLE)/progress_report.tex $(INSTRUCTIONS)/aps360.sty figures/ | $(BUILD)
	$(call prep_source,$(DELIVERABLE)/progress_report.tex,progress_report.tex)
	@test -d figures && cp -r figures $(BUILD)/
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) progress_report.tex
	@cd $(BUILD) && $(LATEX) $(LATEX_FLAGS) progress_report.tex
	@cp $(BUILD)/progress_report.pdf $(DELIVERABLE)/$@

clean:
	rm -rf $(BUILD) $(INSTRUCTIONS)/template.pdf $(DELIVERABLE)/proposal.pdf $(DELIVERABLE)/progress_report.pdf
