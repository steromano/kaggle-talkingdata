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

## BUILD FEATURES 
events <- read_data('events')
app_events <- read_data('app_events')
app_labels <- read_data('app_labels')
label_categories <- read_data('label_categories')

# Generic aggregations of all apps ------------
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
  filter(n_events >= 3) %>%
  select(-n_events)

# App-level features ------------
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
    apps = p_rep(app_id, times_active + 1) %>% paste0(collapse = ',')
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

categories_pca <- 
  device_categories %>%
  select(starts_with('appcat')) %>%
  keep(~ ! all(. == 0)) %>%
  # Not 100% sure we should scale here but it does
  # seem to give better decomposition
  prcomp(scale. = TRUE)

device_categories_pca <-
  bind_cols(
    select(device_categories, -starts_with('appcat')),
    select(device_categories, starts_with('appcat')) %>%
      predict(categories_pca, .) %>%
      as.data.frame %>%
      select(1:15) %>%
      set_names(paste0('appcat_comp', 1:15))
  )

device_events_time <- 
  events %>%
  mutate(tod = paste0('event_tod', hour(timestamp) %/% 4)) %>%
  count(device_id, tod) %>%
  group_by(device_id) %>%
  mutate(event_all = sum(n)) %>%
  ungroup %>%
  spread(tod, n) %>%
  fillna(0)

device_mobility <-
  events %>%
  filter(abs(longitude) + abs(latitude) > 0) %>%
  group_by(device_id) %>%
  mutate(count = n()) %>%
  filter(count >= 5) %>%
  summarise(
    location_var = var(longitude) + var(latitude),
    location_med_lon = median(longitude),
    location_med_lat = median(latitude)
  )

## Model
library(xgboost)
encode_as_integer <- function(x) {
  as.numeric(factor(x, levels = sort(unique(x))))
}

phone_brand_device_model <- 
  read_data('phone_brand_device_model') %>%
  mutate(brand_device = paste0(phone_brand, '-', device_model)) %>%
  # just encode all this stuff as integers
  mutate_each(funs(encode_as_integer), -device_id)
gender_age_train <- 
  read_data('gender_age_train') %>%
  mutate_each(funs(encode_as_integer), gender, group)

train_data <-
  gender_age_train %>%
  inner_join(phone_brand_device_model) %>%
  left_join(device_app_counts) %>%
  left_join(device_categories_pca) %>%
  left_join(device_events_time) %>%
  left_join(device_mobility) %>%
  fillna(-1)

features <- c(
  'device_model', 'phone_brand', 'brand_device',
  paste0('app', 'min_installed', 'max_installed', 'avg_n_active'),
  paste0('appcat_comp', 1:15),
  paste0('event_tod', 0:5), 'event_all',
  paste0('location_', c('var', 'med_lat', 'med_lon'))
)


holdout <- caret::createDataPartition(train_data$group, p = 0.1, list = FALSE)
dtrain <- xgb.DMatrix(
  data = data.matrix(train_data[-holdout, features]), 
  label = train_data$group[-holdout] - 1
)
dval <- xgb.DMatrix(
  data = data.matrix(train_data[holdout, features]), 
  label = train_data$group[holdout] - 1
)
watchlist = list(val = dval, train = dtrain)

preds_to_probs <- function(preds, num_class) {
  preds %>% matrix(nrow = num_class) %>% t
} 

cross_entropy <- function(preds, dtrain) {
  labels <- getinfo(dtrain, 'label')
  probs <- preds_to_probs(preds, num_class = length(unique(labels)))
  n <- length(labels)
  
  ce <- 
    1:n %>%
    map_dbl(~ log(probs[., labels[.] + 1])) %>%
    sum %>%
    multiply_by(- 1 / n)
  
  list(metric = 'cross_entropy', value = ce)
}

params <- list(
  objective = 'multi:softprob',
  num_class = length(unique(getinfo(dtrain, 'label'))),
  booster = 'gbtree',
  max_depth = 10,
  eta = 0.007,
  subsample = 0.9,
  colsample_bytree = 0.6
)

fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  verbose = 1,
  early.stop.round = 100,
  watchlist = watchlist,
  maximize = FALSE,
  feval = cross_entropy
)

saveRDS(fit_xgb, 'RDS/fit_xgb.rds')

library(randomForest)
fit_rf <- randomForest(
  x = train_data[-holdout, features],
  y = as.factor(train_data[-holdout, ]$group),
  ntree = 1000,
  mtry = 5,
  do.trace = TRUE,
  nodesize = round(nrow(train_data) * 0.0005),
  importance = TRUE
)

smooth_probs <- function(probs, gamma = 0.01) {
  probs %>%
    + gamma %>%
    divide_by(rowSums(.))
}

probs <- 
  fit_rf %>%
  predict(train_data[holdout, ], type = 'prob') %>%
  smooth_probs(gamma = 1/3.5)

cross_entropy <- function(probs, actual) {
  n <- nrow(probs)
  1:n %>%
    map_dbl(~ log(probs[., actual[.]])) %>%
    sum %>%
    multiply_by(- 1 / n)
}
cross_entropy(probs, train_data[holdout, ]$group)

## Submission

gender_age_test <- read_data('gender_age_test')

test_data <-
  gender_age_test %>%
  inner_join(phone_brand_device_model) %>%
  left_join(device_app_counts) %>%
  left_join(device_categories_pca) %>%
  left_join(device_events_time) %>%
  left_join(device_mobility) %>%
  fillna(-1)

groupmap <- read_data('gender_age_train') %$% group %>% unique %>% sort
probs <- 
  test_data %>%
  select_(.dots = features) %>%
  data.matrix %>%
  predict(fit_xgb, .) %>%
  preds_to_probs(12) %>%
  data.frame(test_data$device_id, .) %>%
  set_names(c('device_id', groupmap)) %>%
  distinct(device_id, .keep_all = TRUE)

write_csv(probs[, names(read_data('sample_submission'))], 'xgb_submission.csv')
