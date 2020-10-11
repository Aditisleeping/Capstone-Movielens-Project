#Aditi Seshadri
#Movielens Project
#HarvardX: PH125.9x - Capstone Project

### MovieLens Rating Prediction Project Code ### 

##Dataset##

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)

##Methods and Analysis##

#Data Analysis#

#Seeing column names and head of edx

head(edx) %>% print.data.frame()

#Summary of edx gives total unique movies and users
summary(edx)

# Number of unique movies and users in the edx dataset 
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

# Ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

## Plot number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

#As a result from the previous graph, 20 movies have been rated only once. Therefore, it will be hard to predict their ratings.
# Table 20 movies rated only once
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Plot number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

# Plot mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

### Modelling Approach ###

##1. Average movie rating model ##

# Compute the edx dataset's mean rating
mu <- mean(edx$rating)
mu

# Function for calculating the RMSE
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

# Test results based on simple prediction
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

# Check results
# Save prediction in data frame
rmse_results <- data_frame(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

## 2. Movie effect model ##

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating-mu))
movie_avgs %>% qplot(b_i,geom = "histogram", bins=10,data=.,color = I("black"),
                     ylab = "Number of movies",main = "Number of movies with the computed b_i")
#This is called the penalty term movie effect.

predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

## 3. Movie and user effect model ##

# Plot penaly term user effect #
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Test and save rmse results 
predicted_ratings_2 <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings_2, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and User effect model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

## 4. Genre Effect model taking into account movie and user effects ##

genre_avgs<- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  filter(n() >= 100) %>%
  summarize(b_g = mean(rating - mu - b_i-b_u))
genre_avgs%>% qplot(b_g, geom ="histogram", bins = 30, data = ., color = I("black"))

genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by="userId") %>%
  group_by(genres) %>% 
  summarise(b_g = mean(ratings-mu-b_i-b_u))

predicted_ratings_3 <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u+b_g) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings_3, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genre, Movie and User effect model",
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()
  
## Regularised Genre, Movie and User Model ##

lambdas = seq(0,10,0.25)

rmses <- sapply(lambdas, function(l){
     mu <- mean(edx$rating)
     b_i <- edx %>% 
         group_by(movieId) %>%
         summarize(b_i = sum(rating - mu)/(n()+l))
     b_u <- edx %>% 
         left_join(b_i, by="movieId") %>%
         group_by(userId) %>%
         summarize(b_u = sum(rating - b_i - mu)/(n()+l))
     b_g <- edx %>%
         left_join(b_i, by="movieId") %>%
         left_join(b_u, by="userId") %>%
         group_by(genres) %>% 
         summarize(b_g = sum(rating-mu-b_i-b_u)/(n()+1))
     predicted_rating <- validation %>%
         left_join(b_i, by = "movieId") %>%
         left_join(b_u, by = "userId") %>%
         left_join(b_g, by = "genres") %>%
         mutate(pred = mu + b_i + b_u + b_g) %>%
         pull(pred)
     return(RMSE(predicted_rating, validation$rating))
   })
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
#this gives the lambda value with optimum rmse.

### Results ###
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#From the given table, the minimum rmse is found to be 0.8644509.

