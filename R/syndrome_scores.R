library(jsonlite)
library(dplyr)
library(ggplot2)
library(Cairo) # anti-aliased graphics
library(wesanderson)

metrics22 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00022-kid50k_full.jsonl")) %>% 
  data_frame()
metrics22$syndrome <- rep("Cornelia-de-Lange", nrow(metrics22))
metrics22$kimg <- seq(0, 2500, by=500)

metrics23 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00023-kid50k_full.jsonl")) %>% 
  data_frame()
metrics23$syndrome <- rep("HPMRS", nrow(metrics23))
metrics23$kimg <- seq(0, 2500, by=500)

metrics24 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00024-kid50k_full.jsonl")) %>% 
  data_frame()
metrics24$syndrome <- rep("Williams-Beuren", nrow(metrics24))
metrics24$kimg <- seq(0, 2500, by=500)

metrics25 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00025-kid50k_full.jsonl")) %>% 
  data_frame()
metrics25$syndrome <- rep("Kabuki", nrow(metrics25))
metrics25$kimg <- seq(0, 2500, by=500)

metrics26 <- stream_in(file(
  "D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/00026-kid50k_full.jsonl")) %>% 
  data_frame()
metrics26$syndrome <- rep("Angelman", nrow(metrics26))
metrics26$kimg <- seq(0, 2500, by=500)



metrics <- bind_rows(metrics22, metrics23, 
                     metrics24, metrics25,
                     metrics26)

metrics$KID <- metrics$results$kid50k_full

metrics <- metrics %>% select(c("KID", "kimg", "syndrome", "metric"))
metrics$KID[c(1, 7, 13, 19, 25)] <- c(rep(0.02, 5)) # replace 0 kimg KID for nicer plot
metrics$syndrome <- factor(metrics$syndrome, levels = c("Cornelia-de-Lange", "Williams-Beuren", "Kabuki", "Angelman", "HPMRS"))

pal <- c("#00A08A", "#F2300F", "#EBCC2A", "#3B9AB2", "#046C9A")

g <- metrics %>%
  ggplot(aes(x=kimg, y=KID, color=syndrome)) +
  geom_line(size=1) +
  geom_point() +
#  stat_smooth(method="loess", formula=y~x, level=0) +
  coord_cartesian(ylim=c(0, 0.0125)) +
  geom_abline(intercept=min(metrics$KID), slope=0, color = pal[1], linetype=3) +
  theme_minimal() +
  scale_color_manual(values = pal) +
  xlab("Training Iterations [kimg]") +
  ylab("Evaluation Metric [KID]")

g


ggsave(g, filename = 'D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/scores/syndrome_scores.png', 
     dpi = 300, type = 'cairo',
     width = 12, height = 6, units = 'in',
     bg = "transparent")


ggplot(data_frame(name=c("a", "b"), val = c(4, 6)), aes(x=name, y=val, fill=name)) +
  geom_col()
