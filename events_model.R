source('load.R')
preds_to_probs <- function(preds, num_class) {
  preds %>% matrix(nrow = num_class) %>% t
} 

xgb_cross_entropy <- function(preds, dtrain) {
  labels <- getinfo(dtrain, 'label') + 1
  num_class = length(unique(labels))
  probs <- 
    preds_to_probs(preds, num_class = num_class) %>%
    set_colnames(as.character(1:num_class))
  
  list(metric = 'cross_entropy', value = cross_entropy(probs, labels))
}

## ---------------

gender_age_train <- 
  read_raw('gender_age_train') %>%
  mutate_each(funs(encode_as_integer), gender, group)
device_model_features <- read_clean('device_model_features')
device_app_features <- read_clean('device_app_features')
device_event_features <- read_clean('device_event_features')

train_data <-
  gender_age_train %>%
  inner_join(device_model_features) %>%
  inner_join(device_event_features) %>%
  left_join(device_app_features) %>%
  fillna(-1)


features <- c(
  paste0('appcat_comp', 1:30),
  paste0('event_timeslot', c(1, 4, 5)),
  'phone_brand', 'brand_model'
)

# xgboost model
library(xgboost)

holdout <- caret::createDataPartition(
  train_data$group, 
  p = 0.1, 
  list = FALSE
)
dtrain <- xgb.DMatrix(
  data = data.matrix(train_data[-holdout, features]), 
  label = train_data$group[-holdout] - 1
)
dval <- xgb.DMatrix(
  data = data.matrix(train_data[holdout, features]), 
  label = train_data$group[holdout] - 1
)
watchlist = list(val = dval, train = dtrain)

params <- list(
  objective = 'multi:softprob',
  num_class = length(unique(getinfo(dtrain, 'label'))),
  booster = 'gbtree',
  max_depth = 3,
  eta = 0.007,
  subsample = 0.6,
  colsample_bytree = 0.6
)

fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  verbose = 1,
  early.stop.round = 100,
  watchlist = watchlist,
  maximize = FALSE,
  feval = xgb_cross_entropy
)

saveRDS(fit, 'models/xgb_events_fit.rds')

gender_age_test <- read_raw('gender_age_test')

test_data <-
  gender_age_test %>%
  inner_join(device_model_features) %>%
  inner_join(device_event_features) %>%
  left_join(device_app_features) %>%
  fillna(-1)

test_probs <- 
  test_data %>%
  select_(.dots = features) %>%
  data.matrix %>%
  predict(fit_xgb, .) %>%
  preds_to_probs(12) %>%
  data.frame(test_data$device_id, .) %>%
  set_names(c('device_id', sort(groups))) %>%
  group_by(device_id) %>%
  summarise_each(funs(mean))

write_preds(test_probs, 'events')

