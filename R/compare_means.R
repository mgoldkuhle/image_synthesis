# example for a linear model to compare group means
library(ggplot2)
library(dplyr)
library(tidyr)

x <- rep(1, 10)
y1 <- x + 3 + rnorm(length(x), sd=1)
y2 <- x + 5 + rnorm(length(x), sd=1)

lines <- data.frame(sl = c(0,0), 
                 int = c(mean(y1),mean(y2)), 
                 name = c('y1','y2'))

df <- tibble(x, y1, y2) %>% pivot_longer(-x)

df %>%
  ggplot(aes(x=x, y=value)) +
  geom_point(aes(group = name, color = name), show.legend = FALSE) +
  geom_abline(data=lines, aes(slope = sl, intercept = int, colour = name), show.legend = FALSE) +
  ylab("y") +
  scale_x_discrete(labels = NULL) +
  theme_minimal(base_size = 16)
  
  
