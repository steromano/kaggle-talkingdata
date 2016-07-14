library(dplyr)
library(magrittr)
library(purrr)
library(readr)
library(tidyr)
library(stringr)
library(lubridate)

read_data <- function(file) {
  suppressWarnings(
    file.path('data', file) %>%
      paste0('.csv') %>%
      read_csv(col_types = cols(
        app_id = col_character(),
        device_id = col_character()
      )) %>%
      distinct
  )
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

events <- read_data('events')
app_events <- read_data('app_events')
app_labels <- read_data('app_labels')
label_categories <- read_data('label_categories')

device_apps <-
  app_events %>%
  inner_join(select(events, ends_with('_id'))) %>%
  select(device_id, app_id) %>%
  distinct

app_categories <-
  app_labels %>%
  inner_join(label_categories) %>%
  mutate(
    category = paste0('appcat_', clean_names(category, '')),
    .x = 1
  ) %>%
  select(-label_id) %>%
  distinct %>%
  spread(category, .x) %>%
  fillna(0)

device_categories <-
  device_apps %>%
  inner_join(app_categories) %>%
  group_by(device_id) %>%
  summarise_each(funs(sum), starts_with('appcat'))
