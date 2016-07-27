source('load.R')
## Model
library(xgboost)


gender_age_train <- 
  read_raw('gender_age_train') %>%
  mutate_each(funs(encode_as_integer), gender, group)
device_model_features <- read_clean('device_model_features')
device_app_features <- read_clean('device_app_features')
device_event_features <- read_clean('device_event_features')

train_events_data <-
  gender_age_train %>%
  inner_join(device_model_features) %>%
  inner_join(device_app_features) %>%
  inner_join(device_event_features)

features <- c(
  paste0('appcat_comp', 1:30),
  paste0('event_timeslot', c(1, 4, 5)),
  'phone_brand', 'brand_model'
)

holdout <- caret::createDataPartition(
  train_events_data$group, 
  p = 0.1, 
  list = FALSE
)
dtrain <- xgb.DMatrix(
  data = data.matrix(train_events_data[-holdout, features]), 
  label = train_events_data$group[-holdout] - 1
)
dval <- xgb.DMatrix(
  data = data.matrix(train_events_data[holdout, features]), 
  label = train_events_data$group[holdout] - 1
)
watchlist = list(val = dval, train = dtrain)

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

params <- list(
  objective = 'multi:softprob',
  num_class = length(unique(getinfo(dtrain, 'label'))),
  booster = 'gbtree',
  max_depth = 3,
  eta = 0.005,
  subsample = 0.7,
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
  feval = xgb_cross_entropy
)

saveRDS(fit_xgb, 'RDS/fit_xgb.rds')

gender_age_test <- read_raw('gender_age_test')

test_events_data <-
  gender_age_test %>%
  inner_join(device_model_features) %>%
  inner_join(device_event_features) %>%
  inner_join(device_app_features)

groupmap <- read_raw('gender_age_train') %$% group %>% unique %>% sort
test_events_probs <- 
  test_events_data %>%
  select_(.dots = features) %>%
  data.matrix %>%
  predict(fit_xgb, .) %>%
  preds_to_probs(12) %>%
  data.frame(test_events_data$device_id, .) %>%
  set_names(c('device_id', groupmap)) %>%
  distinct(device_id, .keep_all = TRUE)

test_noevent_probs <- 
  read_data('predictions/base') %>%
  filter(! device_id %in% test_events_data$device_id)

probs <- 
  gender_age_test %>%
  inner_join(bind_rows(
    test_events_probs,
    test_noevent_probs
  ))

write_csv(probs, 'predictions/xgb_submission.csv')
