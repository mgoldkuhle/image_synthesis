library(jsonlite)
library(dplyr)
library(ggplot2)
library(Cairo) # anti-aliased graphics

metrics <- tibble(setup=character(), kimg=numeric(), score=numeric())
directory_path <- "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/scores/"
for (file_name in list.files(directory_path)){
  setup <- substr(file_name, 1, 5)
  file_path <- paste0(directory_path, setup, "_metric-kid50k_full.jsonl")
  metrics_tmp <- stream_in(file(
    file_path)) %>% 
    tibble()
  metrics_tmp$setup <- rep(setup, nrow(metrics_tmp))
  seq_len <- nrow(metrics_tmp)
  max_kimg <- as.numeric(substr(metrics_tmp$snapshot_pkl[seq_len], 18, 23))
  metrics_tmp$kimg <- seq(0, max_kimg, length.out=seq_len)
  metrics_tmp$score <- metrics_tmp$results$kid50k_full
  metrics_tmp <- metrics_tmp %>% select(c("score", "kimg", "setup"))
  metrics <- bind_rows(metrics, metrics_tmp)
}

setup_selection <- c("00017", "00073", "00074", "00075", "00076", "00077", "00078")
if (length(setup_selection > 0)) {
  selected_metrics <- metrics %>% filter(setup %in% setup_selection)
} else {
  selected_metrics <- metrics
}

g <- selected_metrics %>%
  ggplot(aes(x=kimg, y=score, color=setup)) +
  geom_line(linewidth=1) +
  geom_point() +
  coord_cartesian(ylim=c(0, 0.015)) +
  geom_abline(intercept=min(selected_metrics$score), slope=0, color = "red", linetype=3) +
  theme_minimal()

g

# setups ranked by best kid score:
# 00017: paper256, gamma=0.4, ada=0.5
# 00021: paper256, gamma=0.6, ada=0.6
# 00012: auto2, gamma=0.4, ada=0.6, generated samples kinda suck
# 00009: paper256, gamma=1, ada=0.5
# 00010: paper256, gamma=1, ada=0.6
# 00020: paper256, gamma=0.8, ada=0.6
# 00011: paper256, gamma=1, ada=0.7
# 00014: auto2, gamma=0.3, ada=0.5
# 00015: auto2, gamma=0.4, ada=0.5

# ggsave(g, filename = 'D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/scores.png', 
#       dpi = 300, type = 'cairo',
#       width = 8, height = 4, units = 'in')

# su = "00021"
# min_score <- metrics %>% filter(setup == su) %>% select(score) %>% min()
# 
# at_kimg <- metrics %>% filter(setup == su) %>% filter(score == min_score) %>% select(kimg)
# 
# last_score <- metrics %>% filter(setup == su) %>% filter(row_number() == n()) %>% select(score)
# 
# print(min_score)
# print(at_kimg)
# print(last_score)
