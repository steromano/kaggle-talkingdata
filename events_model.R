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
  mutate_each(funs(encode_as_integer), gender, group) %>%
  inner_join(read_raw('events') %>% distinct(device_id))
device_model_features <- 
  read_clean('device_model_features') %>%
  select(device_id, device_model, brand_model) %>%
  gather(feature, value, -device_id)
device_app_features <- read_clean('device_app_features')
device_event_features <- read_clean('device_event_features')

features_frame <- 
  bind_rows(
    device_model_features,
    device_app_features,
    device_event_features
  ) %>%
  as_sparse_matrix

train_device_ids <- gender_age_train$device_id
train_labels <- gender_age_train$group
train_data <- features_frame[match(train_device_ids, rownames(features_frame)), ]

features <- colnames(train_data)

# xgboost model
library(xgboost)

holdout <- caret::createDataPartition(
  train_labels,
  p = 0.1,
  list = FALSE
) %>% as.numeric

dtrain <- xgb.DMatrix(
  data = train_data[-holdout, ], 
  label = train_labels[-holdout] - 1
)
dval <- xgb.DMatrix(
  data = train_data[holdout, ], 
  label = train_labels[holdout] - 1
)
watchlist = list(val = dval, train = dtrain)

params <- list(
  objective = 'multi:softprob',
  eval_metric = 'mlogloss',
  num_class = length(unique(getinfo(dtrain, 'label'))),
  booster = 'gbtree',
  max_depth = 8,
  eta = 0.01,
  # lambda = 5,
  # alpha = 2,
  subsample = 0.8,
  colsample_bytree = 0.5
)

fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  verbose = 1,
  early.stop.round = 100,
  watchlist = watchlist,
  maximize = FALSE
)

saveRDS(fit, 'models/xgb_events_fit.rds')

gender_age_test <- read_raw('gender_age_test')

test_device_ids <- gender_age_test$device_id
test_data <- features_frame[match(test_device_ids, rownames(features_frame)), ]

test_probs <- 
  test_data %>%
  predict(fit, .) %>%
  preds_to_probs(12) %>%
  data.frame(test_data$device_id, .) %>%
  set_names(c('device_id', sort(groups))) %>%
  group_by(device_id) %>%
  summarise_each(funs(mean))

write_preds(test_probs, 'events')
