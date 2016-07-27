library(dplyr)
library(magrittr)
library(purrr)
library(readr)
library(tidyr)
library(stringr)
library(lubridate)

read_data <- function(file, ...) {
  suppressWarnings(
      paste0(file, '.csv') %>%
      read_csv(col_types = cols(
        app_id = col_character(),
        device_id = col_character() 
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