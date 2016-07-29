source('load.R')

events <- read_raw('events')

events_by_time <- 
  events %>%
  mutate(hour = paste0('event_hour', hour(timestamp))) %>%
  count(device_id, hour) %>%
  group_by(device_id) %>%
  mutate(event_n = sum(n)) %>%
  ungroup %>%
  spread(hour, n) %>%
  fillna(0) %>%
  mutate_each(funs(. / event_n), starts_with('event_hour'))

events_location <- 
  events %>%
  filter(abs(longitude) + abs(latitude) > 0) %>%
  group_by(device_id) %>%
  summarise(
    event_med_lon = median(longitude),
    event_med_lat = median(latitude),
    event_location_var = var(longitude) + var(latitude)
  )

events_features <- 
  events_by_time %>%
  left_join(events_location) %>%
  fillna(-1) %>%
  gather(feature, value, -device_id)

write_csv(events_features, 'data/clean/device_event_features.csv')
