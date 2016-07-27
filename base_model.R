source('load.R')
inner_join <- function(...) {
  suppressMessages(dplyr::inner_join(...))
}
right_join <- function(...) {
  suppressMessages(dplyr::right_join(...))
}
expand_grid <- function(...) {
  as.tbl(expand.grid(...))
}

# ----------------------------

device_model_features <- 
  read_raw('phone_brand_device_model') %>%
  mutate(brand_model = paste0(phone_brand, '-', device_model)) %>%
  # just encode all this stuff as integers
  mutate_each(funs(encode_as_integer), -device_id)

write_csv(device_model_features, 'data/clean/device_model_features.csv')

gender_age_train <- 
  read_raw('gender_age_train') %>%
  mutate_each(funs(encode_as_integer), gender, group)

train_data <- inner_join(device_model_features, gender_age_train)

preds_grid <- expand_grid(
  phone_brand = unique(device_model_features$phone_brand),
  brand_model = unique(device_model_features$brand_model),
  group = unique(train_data$group)
) %>% inner_join(
  device_model_features %>% distinct(phone_brand, brand_model)
)


# Naive Bayes base model
nb_nofeatures <- function(data, ...) {
  fit <- 
    data %>%
    count(group) %>% 
    mutate(prior = n / sum(n)) %>%
    select(-n) 
  class(fit) <- c('nb_nofeatures', class(fit))
  fit
}

predict.nb_nofeatures <- function(fit, newdata) {
  spread(fit, group, prior)[rep(1, length.out = nrow(newdata)), ]
}


nb_base <- function(data,
                    brand_prior_weight = 40, 
                    model_prior_weight = 40,
                    model_weight = 1) {
  
  prior <- nb_nofeatures(data)
  
  posterior_frame <- function(by, prior_weight) {
    data %>%
      count_(c(by, 'group')) %>%
      ungroup %>%
      right_join(distinct_(preds_grid, by, 'group')) %>%
      fillna(0) %>%
      inner_join(prior) %>%
      group_by_(by) %>%
      mutate(
        # bare_posterior = n / sum(n),
        posterior = (n + prior_weight * prior) / (sum(n) + prior_weight)
      ) %>%
      ungroup %>%
      select_(by, 'group', 'posterior')
  }
  
  brand_posterior <- 
    posterior_frame('phone_brand', brand_prior_weight) %>%
    rename(brand_posterior = posterior)

  model_posterior <- 
    posterior_frame('brand_model', model_prior_weight) %>%
    rename(model_posterior = posterior)
  
  fit <- 
    preds_grid %>%
    inner_join(brand_posterior) %>%
    inner_join(model_posterior) %>%
    mutate(
      pred = (brand_posterior + model_weight * model_posterior) / (1 + model_weight)
    )
  class(fit) <- c('nb_base', class(fit))
  fit
}

predict.nb_base <- function(fit, newdata) {
  newdata %>%
    mutate(id = 1:nrow(.)) %>%
    select(id, phone_brand, brand_model) %>%
    left_join(fit) %>%
    select(id, group, pred) %>%
    spread(group, pred) %>%
    arrange(id) %>%
    select(-id)
}

cv_folds <- function(data, n_folds = 10) {
  n <- nrow(data)
  1:n %>%
    sample(n) %>%
    split(rep(1:n, each = round(n / n_folds), length.out = n))
}

cross_validate <- function(model, data, n_folds = 10, ...) {
  validate_model <- function(fold) {
    train <- data[-fold, ]
    test <- data[fold, ]
    fit <- model(train, ...)
    cross_entropy(predict(fit, test), test$group)
  }
  cv_folds(data, n_folds) %>%
    map_dbl(validate_model) %>%
    mean
}

cv_results <- 
  expand_grid(
    brand_prior_weight = c(40, 45, 50, 55),
    model_prior_weight = c(15, 17.5, 20),
    model_weight = c(1.1, 1.2, 1.3)
  ) %>%
  by_row(
    lift_dl(partial(cross_validate, model = nb_base, data = train_data, n_folds = 10)),
    .collate = 'cols',
    .to = 'score'
  ) %>%
  arrange(score)

fit <- nb_base(
  train_data, 
  brand_prior_weight = 40,
  model_prior_weight = 17.5,
  model_weight = 1.1
)
saveRDS(fit, 'models/nb_base_fit.rds')

test_preds <- 
  read_raw('gender_age_test') %>%
  inner_join(device_model_features) %>%
  anti_join(read_raw('events')) %>%
  bind_cols(predict(fit, .)) %>%
  select(device_id, matches('[0-9]+')) %>%
  group_by(device_id) %>%
  summarise_each(funs(mean)) %>%
  set_names(c('device_id', sort(groups)))

write_preds(test_preds, 'base')



  
