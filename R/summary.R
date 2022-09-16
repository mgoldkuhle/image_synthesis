library(dplyr)
library(readr)
library(ggplot2)

syndromes <- read_tsv("D:/Users/Manu/ownCloud/IGSB/thesis/data/v1_0_2/metadata/gmdb_syndromes_v1.0.2.tsv")

cat(paste0(
  length(unique(syndromes$syndrome_name)), " unique syndromes \n",
  sum(syndromes$num_of_subjects), " unique patients"))

syndromes <- syndromes %>% arrange(desc(num_of_images))
syndromes$id <- as.numeric(row.names(syndromes))

ggplot(syndromes, aes(x=id, y=num_of_images)) +
  geom_bar(stat="identity", fill="#F0CB00") +
  labs(x="Syndrome ID", y="Number of Images") +
  theme_minimal()
