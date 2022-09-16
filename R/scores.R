library(jsonlite)
library(dplyr)
library(ggplot2)
library(Cairo) # anti-aliased graphics

metrics3 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00003-kid50k_full.jsonl")) %>% 
  data_frame()
metrics3$setup <- rep("00003", nrow(metrics3))
metrics3$kimg <- seq(0, 2000, by=400)
# metrics8 <- stream_in(file(
#  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00008-kid50k_full.jsonl")) %>% 
#  data_frame()
# metrics8$setup <- rep("00008", nrow(metrics8))
# metrics8$kimg <- c(2400, 18000)
metrics9 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00009-kid50k_full.jsonl")) %>% 
  data_frame()
metrics9$setup <- rep("00009", nrow(metrics9))
metrics9$kimg <- seq(0, 3000, by=100)
metrics10 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00010-kid50k_full.jsonl")) %>% 
  data_frame()
metrics10$setup <- rep("00010", nrow(metrics10))
metrics10$kimg <- seq(0, 3000, by=100)
metrics11 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00011-kid50k_full.jsonl")) %>% 
  data_frame()
metrics11$setup <- rep("00011", nrow(metrics11))
metrics11$kimg <- seq(0, 3000, by=100)
metrics12 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00012-kid50k_full.jsonl")) %>% 
  data_frame()
metrics12$setup <- rep("00012", nrow(metrics12))
metrics12$kimg <- seq(0, 3000, by=100)
metrics14 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00014-kid50k_full.jsonl")) %>% 
  data_frame()
metrics14$setup <- rep("00014", nrow(metrics14))
metrics14$kimg <- seq(0, 3000, by=100)
metrics15 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00015-kid50k_full.jsonl")) %>% 
  data_frame()
metrics15$setup <- rep("00015", nrow(metrics15))
metrics15$kimg <- seq(0, 3000, by=100)
metrics17 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00017-kid50k_full.jsonl")) %>% 
  data_frame()
metrics17$setup <- rep("00017", nrow(metrics17))
metrics17$kimg <- seq(0, 3000, by=100)
metrics20 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00020-kid50k_full.jsonl")) %>% 
  data_frame()
metrics20$setup <- rep("00020", nrow(metrics20))
metrics20$kimg <- seq(0, 3000, by=100)
metrics21 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00021-kid50k_full.jsonl")) %>% 
  data_frame()
metrics21$setup <- rep("00021", nrow(metrics21))
metrics21$kimg <- seq(0, 3000, by=100)



metrics <- bind_rows(metrics3, metrics9, metrics10, 
                     metrics11, metrics12, metrics14, metrics15,
                     metrics17, metrics20, metrics21)

metrics$score <- metrics$results$kid50k_full

metrics <- metrics %>% select(c("score", "kimg", "setup", "metric"))

g <- metrics %>%
  ggplot(aes(x=kimg, y=score, color=setup)) +
  geom_line(size=1) +
  geom_point() +
#  stat_smooth(method="loess", formula=y~x, level=0) +
  coord_cartesian(ylim=c(0, 0.02)) +
  geom_abline(intercept=min(metrics$score), slope=0, color = "red", linetype=3) +
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
