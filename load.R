library(dplyr)
library(magrittr)
library(purrr)
library(readr)
library(tidyr)
library(tibble)
library(stringr)
library(lubridate)

read_data <- function(file, ...) {
  suppressWarnings(
    paste0(file, '.csv') %>%
      read_csv(col_types = cols(
        app_id = col_character(),
        device_id = col_character(),
        value = col_double()
      ), ...)
  )
}

read_raw <- function(file, ...) {
  read_data(file.path('data/raw', file), ...) %>% distinct
}
read_clean <- function(file, ...) {
  read_data(file.path('data/clean', file, ...))
}

clean_names <- function(x, sep = '_') {
  x %>% str_replace_all(
    '[[:blank:][:punct:]]+',
    sep
  ) %>% tolower
}

fillna <- function(x, fill) {
  x[is.na(x)] <- fill
  x
}

encode_as_integer <- function(x) {
  as.numeric(factor(x, levels = sort(unique(x))))
}

groups <- names(read_raw('sample_submission', n_max = 1))[-1]

write_preds <- function(preds, name) {
  write_csv(
    preds[, c('device_id', groups)], 
    paste0('predictions/', name, '.csv')
  )
}

cross_entropy <- function(probs, labels) {
  labels <- as.character(labels)
  probs <- as.matrix(probs[, sort(unique(labels))])
  1:nrow(probs) %>%
    map_dbl(~ - log(probs[., labels[.]])) %>%
    mean
}

as_sparse_matrix <- function(long_tbl) {
  require(Matrix)
  
  id <- names(long_tbl)[1]
  key <- names(long_tbl)[2]
  value <- names(long_tbl)[3]
  row_nms <- unique(long_tbl[[id]])
  col_nms <- unique(long_tbl[[key]])
  
  sparseMatrix(
    i = match(long_tbl[[id]], row_nms),
    j = match(long_tbl[[key]], col_nms),
    x = long_tbl[[value]],
    dimnames = list(row_nms, col_nms)
  )
}