library(jsonlite)
library(dplyr)
library(ggplot2)

runs <- tibble(
  setup=character(), 
  syndrome=factor(levels=c(0, 1, 2, 3, 12), 
                  labels=c("CdL", "WB", "Kabuki", "Angelman", "HPMRS")),
  kimg=numeric(), 
  gamma=numeric(),
  ada=numeric(),
  last_kid=numeric(),
  best_kid=numeric(),
  best_kid_kimg=numeric(),
  cv_repeat=numeric(),
  cv_fold=numeric(),
  num_images=numeric()
  )

scores_path <- "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/scores/"
options_path <- "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/training_options/"

for (file_name in list.files(scores_path)){
  setup <- substr(file_name, 1, 5)
  score_path <- paste0(scores_path, setup, "_metric-kid50k_full.jsonl")
  metrics_tmp <- stream_in(
    file(score_path)) %>% 
    tibble()
  metrics_tmp$setup <- rep(setup, nrow(metrics_tmp))
  seq_len <- nrow(metrics_tmp)
  max_kimg <- as.numeric(substr(metrics_tmp$snapshot_pkl[seq_len], 18, 23))
  metrics_tmp$kimg <- seq(0, max_kimg, length.out=seq_len)
  metrics_tmp$score <- metrics_tmp$results$kid50k_full
  metrics_tmp <- metrics_tmp %>% select(c("score", "kimg", "setup"))
  
  run_tmp <- tibble(
    setup=character(), 
    syndrome=factor(levels=c(0, 1, 2, 3, 12), 
                    labels=c("CdL", "WB", "Kabuki", "Angelman", "HPMRS")),
    kimg=numeric(), 
    gamma=numeric(),
    ada=numeric(),
    last_kid=numeric(),
    best_kid=numeric(),
    best_kid_kimg=numeric(),
    cv_repeat=numeric(),
    cv_fold=numeric(),
    num_images=numeric()
  ) %>% add_row()
  run_tmp$setup <- metrics_tmp$setup[1]
  run_tmp$kimg <- max_kimg
  run_tmp$last_kid <- metrics_tmp$score[seq_len]
  run_tmp$best_kid <- min(metrics_tmp$score)
  run_tmp$best_kid_kimg <- metrics_tmp$kimg[which.min(metrics_tmp$score)]
  
  option_path <- paste0(options_path, setup, "_training_options.json")
  if (file.exists(option_path)){
    options_tmp <- fromJSON(
      file(option_path)) %>% 
      tibble()
    options_tmp <- options_tmp$.
    run_tmp$gamma <- options_tmp$loss_kwargs$r1_gamma
    run_tmp$ada <- options_tmp$ada_target
    data_path <- options_tmp$training_set_kwargs$path
    data_set <- regmatches(data_path, regexpr("[^/]+(?=\\.zip$)", data_path, perl=TRUE))
    if (data_set %in% c("0", "1", "2", "3", "12")){
      data_set <- factor(data_set, levels=c(0, 1, 2, 3, 12))
    } else {
      data_set <- NA
    }
    run_tmp$syndrome <- data_set
    repetition <- strsplit(data_path, "/")[[1]][8]
    repetition <- substr(repetition, nchar(repetition), nchar(repetition))
    if (!is.na(repetition)){
      if (repetition == "n") {
        repetition <- 1
      } else if (repetition %in% c("2", "3", "4", "5")){
        repetition <- as.numeric(repetition)
      } else {
        repetition <- NA
      }
      run_tmp$cv_repeat <- repetition
    }
    
    
    fold <- strsplit(data_path, "/")[[1]][9]
    if (!is.na(fold)){
      if (fold %in% c("1", "2", "3", "4", "5")){
        fold <- as.numeric(fold)
      } else {
        fold <- NA
      }
      run_tmp$cv_fold <- fold
    }
    
    
    run_tmp$num_images <- options_tmp$training_set_kwargs$max_size
    
  } else {
    print(paste("No options file exists for run", setup))
  }
  print(run_tmp)
  runs <- bind_rows(runs, run_tmp)
}

write.csv(runs, "C:/users/manu/ownCloud/IGSB/thesis/synthesis/results/runs.csv")

# function to display lm result as text in plot. only works with specified columns right now.
lm_eqn <- function(df){
  m <- lm(best_kid ~ num_images, df);
  eq <- substitute(italic(y) == a~-~b %.% italic(x)*","~~italic(r)^2~"="~r2, 
                   list(a = format(unname(coef(m)[1]), digits = 2),
                        b = format(unname(abs(coef(m)[2])), digits = 2),
                        r2 = format(summary(m)$r.squared, digits = 3)))
  as.character(as.expression(eq));
}

g <- runs %>% ggplot(aes(x = num_images, y = best_kid)) +
  geom_point(aes(color = syndrome)) +
  geom_smooth(method = "lm", se = FALSE, colour = "grey", size=0.5) +
#  geom_text(x = 250, y = 0.0095, label = lm_eqn(runs), parse = TRUE, colour = "grey") +
  scale_x_continuous(limits = c(-10, 320)) +
  xlab("Number of Images") + 
  ylab("Best KID") +
  theme_minimal(base_size = 16)
g


model <- runs %>% 
  lm(formula = best_kid ~ num_images, data = .)
summary(model)

anova_model <- runs %>% filter(!is.na(syndrome)) %>%
  lm(best_kid ~ syndrome, data = .)
aov(anova_model)
pairwise.t.test(runs$best_kid, runs$syndrome, p.adj = "none")

h <- runs %>% filter(setup %in% c("00073", "00074", "00075", "00076", "00077", "00078")) %>% 
  ggplot(aes(x = num_images, y = best_kid)) +
  geom_point(colour = "blue") +
  geom_smooth(method = "lm", formula = y ~ x,se = FALSE, colour = "grey", size=0.5) +
  scale_x_continuous(limits = c(-10, 320)) +
  xlab("Number of Images") + 
  ylab("Best KID") +
  theme_minimal(base_size = 16)
h
model <- runs %>% filter(setup %in% c("00073", "00074", "00075", "00076", "00077", "00078")) %>% 
  lm(formula = best_kid ~ num_images, data = .)
summary(model)

i <- runs %>% filter(!is.na(syndrome)) %>% ggplot(aes(x = num_images, y = best_kid_kimg)) +
  geom_point(aes(color = syndrome)) +
  scale_x_continuous(limits = c(-10, 320)) +
  geom_smooth(method = "lm", se = FALSE, colour = "grey", size=0.5) +
  xlab("Number of Images") + 
  ylab("KIMG that scored best KID") +
  theme_minimal(base_size = 16)
i
model <- lm(best_kid_kimg ~ num_images, runs)
summary(model)

j <- runs %>% filter(!is.na(syndrome)) %>% ggplot(aes(x = best_kid_kimg, y = best_kid)) +
  geom_point(aes(color = syndrome)) +
  #scale_x_continuous(limits = c(-10, 320)) +
  geom_smooth(method = "lm", se = FALSE, colour = "grey", size=0.5) +
  xlab("KIMG that scored best KID") + 
  ylab("Best KID") +
  theme_minimal(base_size = 16)
j
model <- lm(best_kid ~ best_kid_kimg, runs)
summary(model)
