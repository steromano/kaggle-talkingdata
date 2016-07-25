source('load.R')

device_model_features <- 
  read_raw('phone_brand_device_model') %>%
  mutate(brand_model = paste0(phone_brand, '-', device_model)) %>%
  # just encode all this stuff as integers
  mutate_each(funs(encode_as_integer), -device_id)

write_csv(phone_brand_device_model, 'data/clean/device_model_features.csv')

gender_age_train <- 
  read_raw('gender_age_train') %>%%>%
  mutate_each(funs(encode_as_integer), gender, agegroup, group)

train_data <- inner_join(device_model_features, gender_age_train)

# Simple Bayesian model
prior <- 
  train_data %>%
  count(group) %>%
  mutate(
    n = n / nrow(train_data)
  ) %>%
  rename(prior = n)

posterior_frame <- function(by, prior_weight) {
  train_data %>%
    count_(c(by, 'group')) %>%
    ungroup %>%
    right_join(
      expand.grid(
        unique(.[[by]]), 1:length(unique(.[['group']]))
      ) %>% set_names(c(by, 'group'))
    ) %>%
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

preds_frame <- function(brand_prior_weight = 10, model_prior_weight = 5) {
  brand_posterior <- 
    posterior_frame('phone_brand', brand_prior_weight) %>%
    rename(brand_posterior = posterior)
  model_posterior <- 
    posterior_frame('brand_model', model_prior_weight) %>%
    rename(model_posterior = posterior)
  train_data %>%
    distinct(phone_brand, brand_model) %>%
    inner_join(brand_posterior) %>%
    inner_join(model_posterior) %>%
    mutate(pred = (brand_posterior + model_posterior) / 2)
}
