# Download + preprocess one TCGA cohort to MOGONET-style CSVs.
# Usage: Rscript download_one_cohort.R COHORT OUT_DIR
#   COHORT in {COADREAD, UCEC, SARC, LUAD}
suppressMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
cohort  <- args[1]
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
work    <- file.path(out_dir, "_gdc")
dir.create(work, recursive = TRUE, showWarnings = FALSE)

projects <- switch(
  cohort,
  COADREAD = c("TCGA-COAD", "TCGA-READ"),
  UCEC     = c("TCGA-UCEC"),
  SARC     = c("TCGA-SARC"),
  LUAD     = c("TCGA-LUAD"),
  stop("Unknown cohort: ", cohort)
)

cat("[", cohort, "] projects: ", paste(projects, collapse = ","), "\n", sep = "")

have_or_make <- function(path, fn) {
  if (file.exists(path)) {
    cat("[skip] already exists: ", basename(path), "\n", sep = "")
    return(readRDS(path))
  }
  obj <- fn()
  saveRDS(obj, path)
  obj
}

# ---- mRNA: STAR-Counts (gene-level) ----
mrna_se <- have_or_make(file.path(out_dir, "mrna_se.rds"), function() {
  q <- GDCquery(project = projects,
                data.category = "Transcriptome Profiling",
                data.type     = "Gene Expression Quantification",
                workflow.type = "STAR - Counts",
                sample.type   = c("Primary Tumor"))
  GDCdownload(q, directory = work, files.per.chunk = 50)
  GDCprepare(q, directory = work, summarizedExperiment = TRUE)
})
cat("[", cohort, "] mRNA: ", ncol(mrna_se), " samples x ", nrow(mrna_se), " genes\n", sep = "")

# ---- methylation: 450K beta values (needs sesameData) ----
meth_se <- have_or_make(file.path(out_dir, "meth_se.rds"), function() {
  q <- GDCquery(project = projects,
                data.category = "DNA Methylation",
                data.type     = "Methylation Beta Value",
                platform      = "Illumina Human Methylation 450",
                sample.type   = c("Primary Tumor"))
  GDCdownload(q, directory = work, files.per.chunk = 30)
  GDCprepare(q, directory = work, summarizedExperiment = TRUE)
})
cat("[", cohort, "] meth: ", ncol(meth_se), " samples x ", nrow(meth_se), " probes\n", sep = "")

# ---- miRNA expression ----
mir_df <- have_or_make(file.path(out_dir, "mirna.rds"), function() {
  q <- GDCquery(project = projects,
                data.category = "Transcriptome Profiling",
                data.type     = "miRNA Expression Quantification",
                sample.type   = c("Primary Tumor"))
  GDCdownload(q, directory = work, files.per.chunk = 50)
  GDCprepare(q, directory = work)
})
cat("[", cohort, "] miRNA: ", ncol(mir_df), " cols\n", sep = "")

# ---- clinical / subtype labels ----
# GDCquery_clinic occasionally fails on multi-project queries with "Supplied
# 261 items ... 610 items of column 'submitter_id'" (data.table assign mismatch).
# Fall back to per-project query and rbind, finally to colData(mrna_se).
clin <- have_or_make(file.path(out_dir, "clinical.rds"), function() {
  out <- tryCatch(GDCquery_clinic(project = projects, type = "clinical"),
                  error = function(e) NULL)
  if (!is.null(out)) return(out)
  message("GDCquery_clinic merged failed; trying per-project")
  parts <- list()
  for (p in projects) {
    parts[[p]] <- tryCatch(GDCquery_clinic(project = p, type = "clinical"),
                           error = function(e) NULL)
  }
  parts <- parts[!sapply(parts, is.null)]
  if (length(parts) > 0) return(do.call(rbind, lapply(parts, as.data.frame)))
  message("All clinical queries failed; using colData(mrna_se) fallback")
  cd <- as.data.frame(colData(mrna_se))
  cd$submitter_id <- substr(rownames(cd), 1, 12)
  cd
})
have_or_make(file.path(out_dir, "molecular_subtype.rds"), function() {
  if (cohort == "COADREAD") {
    coad <- tryCatch(TCGAquery_subtype(tumor = "COAD"), error = function(e) NULL)
    read <- tryCatch(TCGAquery_subtype(tumor = "READ"), error = function(e) NULL)
    parts <- list()
    if (!is.null(coad) && nrow(coad) > 0) parts$coad <- coad
    if (!is.null(read) && nrow(read) > 0) parts$read <- read
    if (length(parts) == 0) return(data.frame())
    common_cols <- Reduce(intersect, lapply(parts, colnames))
    return(do.call(rbind, lapply(parts, function(d) d[, common_cols, drop = FALSE])))
  }
  tum <- switch(cohort, SARC = "SARC", LUAD = "luad", UCEC = "ucec")
  ms <- tryCatch(TCGAquery_subtype(tumor = tum), error = function(e) NULL)
  if (is.null(ms)) data.frame() else ms
})

cat("[", cohort, "] DOWNLOAD DONE\n", sep = "")
