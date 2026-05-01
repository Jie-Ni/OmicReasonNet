# MOGONET-style preprocessing of a downloaded TCGA cohort.
# Usage: Rscript preprocess_cohort.R OUT_DIR [COHORT]
suppressMessages({
  library(SummarizedExperiment)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
# infer cohort from out_dir (last segment) if not given
cohort <- if (length(args) >= 2) args[2] else basename(normalizePath(out_dir))

# Per-cohort label source. "molecular_subtype" uses molecular_subtype.rds with
# given column; "clinical" uses clinical.rds. COADREAD's molecular_subtype is
# Guinney 2015 data covering only ~270 historical patients (4/393 overlap with
# current GDC omics), so we fall back to AJCC stage from clinical (4 classes).
label_source_map <- list(
  COADREAD = list(source = "clinical", col = "ajcc_pathologic_stage", id_col = "submitter_id"),
  UCEC     = list(source = "molecular_subtype", col = "histology_grade",  id_col = "patient"),
  SARC     = list(source = "molecular_subtype", col = "short histo",      id_col = "patient"),
  LUAD     = list(source = "molecular_subtype", col = "expression_subtype", id_col = "patient")
)
ls_cfg <- label_source_map[[cohort]]
label_col   <- ls_cfg$col
patient_col <- ls_cfg$id_col
label_src   <- ls_cfg$source
cat("[preprocess] cohort=", cohort, " label_col=", label_col,
    " (", label_src, ")\n", sep = "")

mrna_se <- readRDS(file.path(out_dir, "mrna_se.rds"))
meth_se <- readRDS(file.path(out_dir, "meth_se.rds"))
mirna   <- readRDS(file.path(out_dir, "mirna.rds"))
clinical <- readRDS(file.path(out_dir, "clinical.rds"))
mol_sub <- if (file.exists(file.path(out_dir, "molecular_subtype.rds")))
           readRDS(file.path(out_dir, "molecular_subtype.rds")) else NULL

# ----- helpers -----
short_bcode <- function(x) substr(x, 1, 12)

# mRNA matrix: rows=genes, cols=samples
mrna_mat  <- assay(mrna_se, "unstranded")
mrna_sym  <- rowData(mrna_se)$gene_name
rownames(mrna_mat) <- paste0(mrna_sym, "|", rowData(mrna_se)$gene_id)
colnames(mrna_mat) <- short_bcode(colnames(mrna_mat))

# meth matrix — keep at cg-probe resolution (matches paper's cg26498301-style IDs)
# Drop probes flagged by sesame masking (low-quality / non-CpG / SNP-overlap).
meth_mat <- assay(meth_se)
rd <- rowData(meth_se)
keep_probe <- rep(TRUE, nrow(meth_mat))
if ("MASK_general" %in% colnames(rd)) {
  keep_probe <- keep_probe & !rd$MASK_general & !is.na(rd$MASK_general)
}
meth_mat <- meth_mat[keep_probe, , drop = FALSE]
# Drop probes with too many NAs across samples
na_frac <- rowMeans(is.na(meth_mat))
meth_mat <- meth_mat[na_frac < 0.10, , drop = FALSE]
# Replace remaining NAs with row means
for (i in seq_len(nrow(meth_mat))) {
  na_i <- is.na(meth_mat[i, ])
  if (any(na_i) && !all(na_i)) {
    meth_mat[i, na_i] <- mean(meth_mat[i, !na_i])
  }
}
meth_gene <- meth_mat   # keep variable name; nodes are cg probes now
colnames(meth_gene) <- short_bcode(colnames(meth_se))
# rownames already cg probe IDs from sesame

# miRNA: extract reads_per_million
mir_rpm_cols <- grep("^reads_per_million_miRNA_mapped_", colnames(mirna), value = TRUE)
mir_mat <- as.matrix(mirna[, mir_rpm_cols])
rownames(mir_mat) <- mirna$miRNA_ID
colnames(mir_mat) <- sub("^reads_per_million_miRNA_mapped_", "", mir_rpm_cols)
colnames(mir_mat) <- short_bcode(colnames(mir_mat))

# ----- align samples by barcode (intersection across 3 modalities) -----
common <- Reduce(intersect, list(colnames(mrna_mat), colnames(meth_gene), colnames(mir_mat)))
cat("common samples: ", length(common), "\n", sep = "")
mrna_mat <- mrna_mat[, common, drop = FALSE]
meth_gene <- meth_gene[, common, drop = FALSE]
mir_mat <- mir_mat[, common, drop = FALSE]

# ----- labels -----
load_labels <- function() {
  if (label_src == "clinical") {
    clin <- readRDS(file.path(out_dir, "clinical.rds"))
    pid <- short_bcode(as.character(clin[[patient_col]]))
    lab <- as.character(clin[[label_col]])
    if (cohort == "COADREAD" && label_col == "ajcc_pathologic_stage") {
      # collapse substages: Stage IA -> Stage I, Stage IIA/B/C -> Stage II, etc.
      lab <- sub("^Stage IA?$",   "Stage I",   lab)
      lab <- sub("^Stage II[ABC]?$","Stage II",  lab)
      lab <- sub("^Stage III[ABC]?$","Stage III", lab)
      lab <- sub("^Stage IV[AB]?$", "Stage IV",  lab)
    }
    data.frame(pid = pid, label = lab, stringsAsFactors = FALSE)
  } else {
    ms <- readRDS(file.path(out_dir, "molecular_subtype.rds"))
    if (!(label_col %in% colnames(ms)) || !(patient_col %in% colnames(ms)))
      stop("label_col / patient_col not in molecular_subtype: ", label_col, " / ", patient_col)
    pid <- short_bcode(as.character(ms[[patient_col]]))
    lab <- as.character(ms[[label_col]])
    data.frame(pid = pid, label = lab, stringsAsFactors = FALSE)
  }
}
labels_df <- load_labels()
# Drop NA / empty / "Notassigned" / "[Not Available]" / "unk" labels
bad <- is.na(labels_df$label) | labels_df$label == "" |
       labels_df$label %in% c("Notassigned", "[Not Available]", "unknown",
                              "[unknown]", "Unknown", "NA", "unk", "[Unknown]")
labels_df <- labels_df[!bad, ]
labels_df <- labels_df[!duplicated(labels_df$pid), ]
labels_df <- labels_df[labels_df$pid %in% common, ]
common    <- intersect(common, labels_df$pid)
mrna_mat  <- mrna_mat[, common, drop = FALSE]
meth_gene <- meth_gene[, common, drop = FALSE]
mir_mat   <- mir_mat[, common, drop = FALSE]
labels_df <- labels_df[match(common, labels_df$pid), ]
# Drop classes with < 5 samples (not enough for stratified 5-fold CV)
tab <- table(labels_df$label)
keep_cls <- names(tab)[tab >= 5]
keep_idx <- labels_df$label %in% keep_cls
labels_df <- labels_df[keep_idx, ]
common    <- common[keep_idx]
mrna_mat  <- mrna_mat[, common, drop = FALSE]
meth_gene <- meth_gene[, common, drop = FALSE]
mir_mat   <- mir_mat[, common, drop = FALSE]
labels_df$label_int <- as.integer(factor(labels_df$label)) - 1L

cat("class table:\n"); print(table(labels_df$label))
cat("final n =", nrow(labels_df), "; n_classes =", length(unique(labels_df$label_int)), "\n")

# ----- transforms -----
mrna_log  <- log2(mrna_mat + 1)
meth_asin <- asin(sqrt(pmin(pmax(meth_gene, 0), 1)))
mir_log   <- log2(mir_mat + 1)

# ----- ANOVA F-test feature selection -----
anova_fsel <- function(M, y, top_n) {
  # M: features x samples; y: integer
  # Use simple F-stat per row across groups
  groups <- split(seq_along(y), y)
  if (length(groups) < 2) return(seq_len(min(top_n, nrow(M))))
  ovr_mean <- rowMeans(M, na.rm = TRUE)
  total_n  <- ncol(M)
  numer <- rep(0, nrow(M))
  denom <- rep(0, nrow(M))
  for (g in groups) {
    if (length(g) < 2) next
    sub <- M[, g, drop = FALSE]
    gmean <- rowMeans(sub, na.rm = TRUE)
    numer <- numer + length(g) * (gmean - ovr_mean)^2
    denom <- denom + rowSums((sub - gmean)^2, na.rm = TRUE)
  }
  k <- length(groups)
  F <- (numer / (k - 1)) / (denom / (total_n - k) + 1e-12)
  F[is.na(F)] <- 0
  order(F, decreasing = TRUE)[seq_len(min(top_n, nrow(M)))]
}

y <- labels_df$label_int
sel_mrna <- anova_fsel(mrna_log,  y, 2000)
sel_meth <- anova_fsel(meth_asin, y, 2000)
sel_mir  <- anova_fsel(mir_log,   y, 500)

# Apply selection
mrna_sel  <- t(mrna_log[sel_mrna, , drop = FALSE])
meth_sel  <- t(meth_asin[sel_meth, , drop = FALSE])
mir_sel   <- t(mir_log[sel_mir, , drop = FALSE])

# ----- stratified 70/30 split -----
set.seed(42)
tr_idx <- unlist(lapply(split(seq_along(y), y), function(g) {
  n <- length(g); sample(g, ceiling(0.7 * n))
}))
te_idx <- setdiff(seq_along(y), tr_idx)

write_no_header <- function(M, path) write.table(M, file = path, sep = ",",
                                                 row.names = FALSE, col.names = FALSE,
                                                 quote = FALSE)

write_no_header(mrna_sel[tr_idx, , drop = FALSE], file.path(out_dir, "1_tr.csv"))
write_no_header(mrna_sel[te_idx, , drop = FALSE], file.path(out_dir, "1_te.csv"))
write_no_header(rownames(mrna_log)[sel_mrna],     file.path(out_dir, "1_featname.csv"))

write_no_header(meth_sel[tr_idx, , drop = FALSE], file.path(out_dir, "2_tr.csv"))
write_no_header(meth_sel[te_idx, , drop = FALSE], file.path(out_dir, "2_te.csv"))
write_no_header(rownames(meth_asin)[sel_meth],    file.path(out_dir, "2_featname.csv"))

write_no_header(mir_sel[tr_idx, , drop = FALSE], file.path(out_dir, "3_tr.csv"))
write_no_header(mir_sel[te_idx, , drop = FALSE], file.path(out_dir, "3_te.csv"))
write_no_header(rownames(mir_log)[sel_mir],      file.path(out_dir, "3_featname.csv"))

write_no_header(y[tr_idx], file.path(out_dir, "labels_tr.csv"))
write_no_header(y[te_idx], file.path(out_dir, "labels_te.csv"))

cat("written: ", out_dir, "\n", sep = "")
cat("class map:\n"); print(levels(factor(labels_df$label)))
