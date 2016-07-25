source('load.R')

events <- read_raw('events')

events_by_time <- 
  events %>%
  mutate(timeslot = paste0('event_timeslot', hour(timestamp) %/% 4)) %>%
  count(device_id, timeslot) %>%
  group_by(device_id) %>%
  mutate(event_n = sum(n)) %>%
  ungroup %>%
  spread(timeslot, n) %>%
  fillna(0) %>%
  mutate_each(funs(. / event_n), starts_with('event_timeslot'))

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
  fillna(-1)

write_csv(events_features, 'data/clean/device_event_features.csv')
