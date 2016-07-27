source('load.R')

base_preds <- read_data('predictions/base')
events_preds <- read_data('predictions/events')
test_data <- read_raw('gender_age_test')

test_data %>%
  select(device_id) %>%
  inner_join(bind_rows(
    base_preds, events_preds
  )) %>%
  write_preds('submission')