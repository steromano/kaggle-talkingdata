source('load.R')
## Model
library(xgboost)

phone_brand_device_model <- 
  read_raw('phone_brand_device_model') %>%
  mutate(brand_device = paste0(phone_brand, '-', device_model)) %>%
  # just encode all this stuff as integers
  mutate_each(funs(encode_as_integer), -device_id)
gender_age_train <- 
  read_raw('gender_age_train') %>%
  mutate_each(funs(encode_as_integer), gender, group)
device_app_features <- read_clean('device_app_features')
device_event_features <- read_clean('device_event_features')

train_data <-
  gender_age_train %>%
  left_join(phone_brand_device_model) %>%
  left_join(device_app_features) %>%
  left_join(device_event_features) %>%
  fillna(-1)

features <- c(
  paste0('appcat_comp', 1:30),
  paste0('event_timeslot', c(1, 4, 5)),
  'brand_device', 'device_model'
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
  max_depth = 4,
  eta = 0.01,
  subsample = 0.5,
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

gender_age_test <- read_raw('gender_age_test')

test_data <-
  gender_age_test %>%
  inner_join(phone_brand_device_model) %>%
  left_join(device_event_features) %>%
  left_join(device_app_features) %>%
  fillna(-1)

groupmap <- read_raw('gender_age_train') %$% group %>% unique %>% sort
probs <- 
  test_data %>%
  select_(.dots = features) %>%
  data.matrix %>%
  predict(fit_xgb, .) %>%
  preds_to_probs(12) %>%
  data.frame(test_data$device_id, .) %>%
  set_names(c('device_id', groupmap)) %>%
  distinct(device_id, .keep_all = TRUE)

write_csv(probs[, names(read_raw('sample_submission'))], 'xgb_submission.csv')
