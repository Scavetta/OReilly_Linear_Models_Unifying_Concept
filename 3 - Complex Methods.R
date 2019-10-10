# Session 4 - Complex Methods
# 
# Lecture: Analytical versus reiterative approaches to minimize the loss function.
# Discussion: Linear models as the basis for advanced methods.
# Exercise: Executing advanced methods in R

#
library(keras)

boston <- read.csv("./data/boston_keras.csv")
summary(boston)

index <- 1:404
training <- boston[index,]
testing <- boston[-index,]

# Z-score transformations:
# Using dplyr to keep data frame structure:
training %>% 
  mutate_at(vars(-MEDV), scale) -> training
testing %>% 
  mutate_at(vars(-MEDV), scale) -> testing


# fit the model:
fit_lm <- lm(MEDV ~ ., data = training)

# plot the coefficients:
data.frame(coef = round(fit_lm$coefficients,2)) %>% 
  rownames_to_column() %>% 
  rename(variable = rowname) %>% 
  filter(variable != "(Intercept)") %>%  
  arrange(coef) %>% 
  mutate(variable = as_factor(variable)) %>% 
  ggplot(aes(coef, variable)) +
  geom_vline(xintercept = 0, col = "dark red", linetype = 2) +
  geom_point() +
  scale_x_continuous("r", limits = c(-4.2,4.2), expand = c(0,0))

#predict on test set
pred_lm <- predict(fit_lm, newdata = testing)
MAE_lm <- sum(abs(pred_lm - testing$MEDV))/102


### Deep Learning, basic regression:
library(keras)

# Data Preparation ----------------------------------------------------------
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset_boston_housing()

## Prepare the training data:
mean <- colMeans(train_data) # mean of each column
std <- apply(train_data, 2, sd) # stdev of each column
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


# Define & Compile the Model ------------------------------------------------
network <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = 13) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1)

# Compile model
network %>% 
  compile(
    optimizer = optimizer_rmsprop(0.001), 
    loss = "mse", 
    metrics = "mae"
  )

# Training & Evaluation ----------------------------------------------------
history <-  network %>% 
  fit(
    train_data,
    train_targets,
    epochs = 60,
    batch_size = 128,
    verbose = FALSE
  )

plot(history)

score <- network %>% evaluate(
  test_data, test_targets
)

cat('Test loss:', score$loss, '\n')
cat('Test MAE:', score$mae, '\n')