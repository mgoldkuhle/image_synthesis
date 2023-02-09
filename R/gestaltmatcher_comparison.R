# compare test set evaluation metrics for gestaltmatcher setups
library(dplyr)
library(caret)
library(ggplot2)
library(lme4)
library(lmerTest)

# metrics produced test_set_evaluation.py
scores <- read_csv("C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/test_accuracies/metrics.csv")

scores$fold <- factor(scores$cv)
scores$method <- factor(scores$setup)

scores  %>%
  ggplot(aes(x=method, y=accuracy)) +
  geom_point() +
  geom_line(aes(group=fold, color=fold), show.legend = FALSE) +
  scale_y_continuous(limits = c(0.2, 1)) +
  ylab("") +
  xlab("") +
#  scale_x_discrete(labels=c("synthetic", "original", "original + weighted synth")) +
  theme_minimal(base_size=20)


tests <- tibble(
  method = character(),
  metric = character(),
  coeff = numeric(),
  p_value = numeric()
)

model <- scores %>% filter(method == "orig" | method == "augmented") %>%
  lmer(formula = sensitivity_0 ~ method + (1|fold), data = .)
model_summ <- summary(model)
model_summ

wilcox_test <- wilcox.test(
  scores%>%filter(method=="orig") %>% select(sensitivity_12) %>% as.matrix %>% as.vector(),
  scores%>%filter(method=="justweightedsynth") %>% select(sensitivity_12) %>% as.matrix %>% as.vector(),
  paired = TRUE
)

tests_tmp <- tibble(
 method = "justweightedsynth",
 metric = "sensitivity_12",
 coeff = model_summ$coefficients[2, 1],
 p_value = model_summ$coefficients[2, 5],
 wilcox_p_value = wilcox_test$p.value
)
tests <- bind_rows(tests, tests_tmp)

write.csv(tests, "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/tests.csv")
tests <- read.csv("C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/tests.csv")
