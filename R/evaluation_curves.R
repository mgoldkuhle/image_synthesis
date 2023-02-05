library(dplyr)
library(readr)
library(ggplot2)
library(lmerTest)


acc_curve <- tibble()

directory_path <- "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/top_acc/"

for (file_name in list.files(directory_path)){
  file_path <- paste0(directory_path, file_name)
  temp_curve <- read_csv(file_path)
  temp_curve$idx <- as.numeric(row.names(temp_curve))
  setup <- sub(paste0("\\.", tools::file_ext(file_path)), "", file_name)
  temp_curve$setup <- setup
  method <- strsplit(setup, "_")[[1]][length(strsplit(setup, "_")[[1]])]
  temp_curve$method <- method
  acc_curve <- bind_rows(acc_curve, temp_curve)
  
}

methods <- acc_curve %>% select(method) %>% unique()
max_accs <- tibble(method = methods$method, max_acc = numeric(length(methods)))
for(method_name in max_accs$method){
  max_accs$max_acc[max_accs$method == method_name] <- acc_curve %>% filter(method == method_name) %>% select(Value) %>% max() %>% round(4)
}

accs_vs <- acc_curve %>% filter(method == "orig" | method == "weightedsynth")

g <- accs_vs %>%
  ggplot(aes(x = idx, y = Value, color = method)) +
  geom_line(alpha=0.2, show.legend = FALSE) +
  geom_smooth(span = 0.3, alpha=0.2, method = 'loess', show.legend = FALSE) +
  scale_y_continuous(limits = c(0.6, 0.99)) +
#  stat_summary(fun.data ="mean_sdl", fun.args = list(mult=1), geom = "smooth") +
  xlab("Epochs") +
  ylab("Accuracy") +
# ggtitle("GestaltMatcher Accuracy with and without added synthetic data") +
  theme_minimal(base_size = 20)
g

model <- lmer(Value ~ method + idx + (1|idx), data = accs_vs)
summary(model)

accs_aggregate <- aggregate(Value ~ idx + method, data = accs_vs, mean)
accs_sd <- aggregate(Value ~ idx + method, data = accs_vs, sd)
accs_aggregate$sd <- accs_sd$Value

ggplot(data = accs_aggregate, aes(x = idx, group = method)) + 
  geom_line(aes(y = Value, color = method), size = 1, show.legend = FALSE) + 
  geom_ribbon(aes(y = Value, ymin = Value - sd, ymax = Value + sd, fill = method), alpha = .2, show.legend = FALSE) +
  scale_y_continuous(limits = c(0.6, 0.99)) +
  xlab("Epochs") + 
  ylab("Accuracy") +
  theme_minimal(base_size = 20)
  #theme(legend.key = element_blank()) + 
  #theme(plot.margin=unit(c(1,3,1,1),"cm"))+
  #theme(legend.position = c(1.1,.6), legend.direction = "vertical") +
  #theme(legend.title = element_blank())

model <- lmer(Value ~ method + idx + (1|idx), data = accs_aggregate)
summary(model)



accs_vs <- accs_vs %>% group_by(method) %>%
  mutate(cum_max_acc = cummax(rev(Value)))
h <- accs_vs %>%
  ggplot(aes(x = idx, y = cum_max_acc, color = method)) +
  geom_smooth(span = 0.05, alpha=0, method = 'loess') +
  xlab("Epochs") + 
  ylab("Cumulative Maximum Accuracy") +
  theme_minimal(base_size = 16)
h
