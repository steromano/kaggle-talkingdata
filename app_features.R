source('load.R')

events <- read_raw('events')
app_events <- read_raw('app_events') %>% inner_join(select(events, ends_with('_id')))
app_labels <- read_raw('app_labels')
label_categories <- read_raw('label_categories')

# Top level: generic apps aggregations
device_app_counts <-
  app_events %>%
  group_by(device_id, event_id) %>%
  summarise(n_installed = n(), n_active = sum(is_active)) %>%
  group_by(device_id) %>%
  summarise(
    n_events = n(),
    app_min_installed = min(n_installed),
    app_max_installed = max(n_installed),
    app_avg_n_active = mean(n_active)
  ) %>%
  select(-n_events) %>%
  gather(feature, value, -device_id)

# Category level: bag of categories 
# (+ PCA?)
device_apps <-
  app_events %>%
  group_by(device_id, app_id) %>%
  summarise(times_active = sum(is_active)) %>%
  ungroup

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

# Experimental: upweight apps that were actually active rather than
# just installed
active_coeff <- 3
device_bag_of_categories <-
  device_apps %>%
  inner_join(app_categories) %>%
  group_by(device_id) %>%
  summarise_each(funs(
    sum(. * (active_coeff - (active_coeff - 1) * exp(- 0.1 * times_active)))
    # sum
    ), starts_with('appcat')) %>%
  gather(feature, value, -device_id)

# Now reduce the 471 dimensional categories space to a few principal components
# categories_pca <- 
#   device_categories %>%
#   select(starts_with('appcat')) %>%
#   keep(~ ! all(. == 0)) %>%
#   # Not 100% sure we should scale here but it does
#   # seem to give better decomposition
#   prcomp(scale. = TRUE)
# 
# ncomp <- 30
# device_categories_pca <-
#   bind_cols(
#     select(device_categories, -starts_with('appcat')),
#     select(device_categories, starts_with('appcat')) %>%
#       predict(categories_pca, .) %>%
#       as.data.frame %>%
#       select(1:ncomp) %>%
#       set_names(paste0('appcat_comp', 1:ncomp))
#   )

# App level: bag of apps
# p_rep <- function(x, each) {
#   map2(x, each, rep) %>% unlist
# }
library(FeatureHashing)
library(slam)
device_apps_string <- 
  device_apps %>%
  group_by(device_id) %>%
  summarise(apps_string = paste0(app_id, collapse = ','))

device_bag_of_apps <- 
  device_apps_string %>%
  hashed.model.matrix(~ split(apps_string, delim = ',', type = 'tf-idf'), . , 2^14) %>%
  set_rownames(device_apps_string$device_id) %>%
  set_colnames(paste0('apphash_', 1:ncol(.))) %>%
  as.simple_triplet_matrix %>% {
    data_frame(
      device_id = .$dimnames[[1]][.$i],
      feature = .$dimnames[[2]][.$j],
      value = .$v
    )
  } %>%
  filter(value > 0)

device_app_features <- bind_rows(
  device_app_counts,
  device_bag_of_apps,
  device_bag_of_categories
)
 
write_csv(device_app_features, 'data/clean/device_app_features.csv') 

