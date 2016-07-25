source('load.R')

events <- read_raw('events')
app_events <- read_raw('app_events')
app_labels <- read_raw('app_labels')
label_categories <- read_raw('label_categories')

# Top level: generic apps aggregations
device_app_counts <- 
  app_events %>%
  inner_join(select(events, ends_with('_id'))) %>%
  group_by(device_id, event_id) %>%
  summarise(n_installed = n(), n_active = sum(is_active)) %>%
  group_by(device_id) %>%
  summarise(
    n_events = n(),
    app_min_installed = min(n_installed),
    app_max_installed = max(n_installed),
    app_avg_n_active = mean(n_active)
  ) %>%
  mutate(
    app_avg_n_active = if_else(n_events > 3, app_avg_n_active, -1)
  ) %>%
  select(-n_events)

# Category level: bag of categories + PCA
# *this works very well and gives my best features. Consider using more*
device_apps <-
  app_events %>%
  inner_join(select(events, ends_with('_id'))) %>%
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

# Count of app categories for each device.
# Experimental: upweight apps that were actually active rather than
# just installed
active_coeff <- 3
device_categories <-
  device_apps %>%
  inner_join(app_categories) %>%
  group_by(device_id) %>%
  summarise_each(funs(sum(
    . * (active_coeff - (active_coeff - 1) * exp(- 0.1 * times_active))
  )), starts_with('appcat'))

# Now reduce the 471 dimensional categories space to a few principal components
categories_pca <- 
  device_categories %>%
  select(starts_with('appcat')) %>%
  keep(~ ! all(. == 0)) %>%
  # Not 100% sure we should scale here but it does
  # seem to give better decomposition
  prcomp(scale. = TRUE)

ncomp <- 30
device_categories_pca <-
  bind_cols(
    select(device_categories, -starts_with('appcat')),
    select(device_categories, starts_with('appcat')) %>%
      predict(categories_pca, .) %>%
      as.data.frame %>%
      select(1:ncomp) %>%
      set_names(paste0('appcat_comp', 1:ncomp))
  )

# App level: bag of apps + LDA
# *this is basically rubbish. Condsider getting rid of this entirely*
device_apps <-
  app_events %>%
  inner_join(select(events, ends_with('_id'))) %>%
  group_by(device_id, app_id) %>%
  summarise(times_active = sum(is_active)) %>%
  ungroup

p_rep <- function(x, each) {
  map2(x, each, rep) %>% unlist
}

library(FeatureHashing)
library(slam)
library(tm)
library(topicmodels)
device_bag_of_apps <- 
  device_apps %>%
  group_by(device_id) %>%
  summarise(
    # Put more weights on active apps by repeating active apps more then 
    # once. This is sort of experimental
    apps = p_rep(app_id, 1 + as.integer(times_active > 0)) %>% paste0(collapse = ',')
  )

apps_lda <- 
  device_bag_of_apps %>%
  hashed.model.matrix(~ split(apps, delim = ',') - 1, ., 2^14) %>%
  as.matrix %>% 
  as.DocumentTermMatrix(weighting = weightTf) %>%
  LDA(k = 10)

device_apps_lda <-
  data.frame(
    device_id = device_bag_of_apps$device_id,
    posterior(apps_lda)$topics %>% set_colnames(paste0('app_topic', 1:10))
  )

device_app_features <-
  # device_apps_lda %>%
  # inner_join(device_categories_pca) %>%
  device_categories_pca %>%
  left_join(device_app_counts) %>%
  fillna(-1)

write_csv(device_app_features, 'data/clean/device_app_features.csv')
