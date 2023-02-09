library(jsonlite)
library(dplyr)
library(ggplot2)

runs <- tibble(
  setup=character(), 
  syndrome=factor(levels=c(0, 1, 2, 3, 12), 
                  labels=c("CdL", "WB", "Kabuki", "Angelman", "HPMRS")),
  kimg=numeric(),
  kid=numeric(),
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
  metrics_tmp$kid <- metrics_tmp$results$kid50k_full
  #metrics_tmp <- metrics_tmp %>% select(c("kid", "kimg", "setup")) %>%
  #  filter(kimg %in% c(0, 500, 1000, 1500, 2000, 2500, 3000))
  
  option_path <- paste0(options_path, setup, "_training_options.json")
  if (file.exists(option_path)){
    options_tmp <- fromJSON(
      file(option_path)) %>% 
      tibble()
    options_tmp <- options_tmp$.
    data_path <- options_tmp$training_set_kwargs$path
    data_set <- regmatches(data_path, regexpr("[^/]+(?=\\.zip$)", data_path, perl=TRUE))
    if (data_set %in% c("0", "1", "2", "3", "12")){
      data_set <- factor(data_set, levels=c(0, 1, 2, 3, 12))
    } else {
      data_set <- NA
    }
    metrics_tmp <- metrics_tmp %>% mutate(syndrome = data_set)
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
      metrics_tmp <- metrics_tmp %>% mutate(cv_repeat = repetition)
    }
    
    
    fold <- strsplit(data_path, "/")[[1]][9]
    if (!is.na(fold)){
      if (fold %in% c("1", "2", "3", "4", "5")){
        fold <- as.numeric(fold)
      } else {
        fold <- NA
      }
      metrics_tmp <- metrics_tmp %>% mutate(cv_fold = fold)
    }
    
    
    metrics_tmp <- metrics_tmp %>% mutate(num_images = options_tmp$training_set_kwargs$max_size)
    
  } else {
    print(paste("No options file exists for run", setup))
  }
  print(metrics_tmp)
  runs <- bind_rows(runs, metrics_tmp)
}

runs <- runs %>% filter(!is.na(syndrome))
run_means <- runs %>% filter(!is.na(syndrome)) %>% group_by(kimg, syndrome) %>% summarise(kid_mean = mean(kid))

run_means %>%
  ggplot(aes(x = kimg, y = kid_mean)) +
  geom_line(aes(color = syndrome))

runs %>% 
  ggplot(aes(x = kimg, y = kid)) +
  geom_smooth(aes(color = syndrome), alpha = 0, show.legend = FALSE, span=0.34) +
  coord_cartesian(ylim=c(0, 0.035)) +
  xlab("Epochs") +
  ylab("KID") +
  theme_minimal(base_size = 16)



