library(ggplot2)
library(ggthemes)
library(dplyr)
library(dotenv)


# Setting project
env_path <- file.path(path.expand("~"), "dotfiles/irpd_configs.env")
dotenv::load_dot_env(env_path)
data_dir <- file.path(Sys.getenv("PROJECT_DIRECTORY"), "data/output/")
plot_dir <- file.path(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots")


# Define configurations and treatments
configs <- list(
  c("test_15", "test_24"), c("test_16", "test_25"), c("test_17", "test_26"))
treatments <- list(
  c("test_12", "test_21"), c("test_13", "test_22"))

df.sim <- read.csv(paste0(data_dir, "cat_sims_t12-t27.csv")) %>%
  mutate(configs = case_when(test %in% configs[[1]] ~ "res1",
                             test %in% configs[[2]] ~ "res2",
                             test %in% configs[[3]] ~ "res3",
                             TRUE ~ "base"),
         treatment = case_when(test %in% treatments[[1]] ~ "Noise",
                               test %in% treatments[[2]] ~ "No-noise",
                               TRUE ~ "Merged"))
df.unames <- read.csv(paste0(data_dir, "cat_names_t12-t27.csv")) %>%
  mutate(configs = case_when(test %in% configs[[1]] ~ "res1",
                             test %in% configs[[2]] ~ "res2",
                             test %in% configs[[3]] ~ "res3",
                             TRUE ~ "base"),
         treatment = case_when(test %in% treatments[[1]] ~ "Noise",
                               test %in% treatments[[2]] ~ "No-noise",
                               TRUE ~ "Merged"))


# Plot Theme
plot_thm <- theme_wsj(base_size = 16, color = "white") +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    axis.title.x = element_blank(),
    strip.text = element_text(family = "mono", size = 18, hjust = 0),
    plot.title = element_text(family = "mono", size = 20, hjust = 0)
  )


# Stage 1 data
subdf.sim.1 <- df.sim %>%
  filter(test >= "test_12" & test <= "test_14",
         method == "cosine") %>%
  group_by(instance, treatment, type) %>%
  summarise(sim_sd = sd(sim, na.rm = T),
            sim = mean(sim, na.rm = T), 
            .groups = "drop")

subdf.sim.2 <- df.sim %>%
  filter(test >= "test_14" & test <= "test_17",
         method == "cosine") %>%
  group_by(instance, configs, type) %>%
  summarise(sim_sd = sd(sim, na.rm = T),
            sim = mean(sim, na.rm = T), 
            .groups = "drop")

subdf.sim.3 <- df.sim %>%
  filter(test >= "test_21" & test <= "test_23",
         method == "cosine") %>%
  group_by(instance, treatment, type) %>%
  summarise(sim_sd = sd(sim, na.rm = T),
            sim = mean(sim, na.rm = T), 
            .groups = "drop")

subdf.sim.4 <- df.sim %>%
  filter(test >= "test_23" & test <= "test_26",
         method == "cosine") %>%
  group_by(instance, configs, type) %>%
  summarise(sim_sd = sd(sim, na.rm = T),
            sim = mean(sim, na.rm = T), 
            .groups = "drop")

subdf.unames.1 <- df.unames %>%
  filter(test >= "test_12" & test <= "test_14") %>%
  group_by(instance, configs, treatment)

subdf.unames.2 <- df.unames %>%
  filter(test >= "test_14" & test <= "test_17") %>%
  group_by(instance, configs, treatment)

subdf.unames.3 <- df.unames %>%
  filter(test >= "test_21" & test <= "test_23") %>%
  group_by(instance, configs, treatment)

subdf.unames.4 <- df.unames %>%
  filter(test >= "test_23" & test <= "test_26") %>%
  group_by(instance, configs, treatment)


# Plots
plot_1 <- ggplot(subdf.sim.1, aes(x = instance, y = sim, fill = treatment)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.5) +
  geom_errorbar(aes(ymin = sim - sim_sd, ymax = sim + sim_sd), 
                width = 0.2, linewidth = 0.8, 
                position = position_dodge(width = 0.6)) +
  labs(y = "Cosine Similarity", fill = "Treatment",
       title = "Stage 1 Categories (Tests 12 - 14)") +
  facet_wrap(~type, labeller = labeller(type = c(
    "definition" = "Category Definition",
    "name" = "Category Name"
  ))) +
  scale_y_continuous(breaks = seq(0, 1, by=0.25)) +
  scale_x_discrete(labels = c("Ucoop", "Udef")) +
  scale_fill_wsj("colors6") +
  plot_thm

plot_2 <- ggplot(subdf.sim.2, aes(x = instance, y = sim, fill = configs)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.5) +
  geom_errorbar(aes(ymin = sim - sim_sd, ymax = sim + sim_sd), 
                width = 0.2, linewidth = 0.8, 
                position = position_dodge(width = 0.6)) +
  labs(y = "Cosine Similarity", x = "Instance", fill = "",
       title = "Stage 1 Categories (Tests 14 - 17)") +
  facet_wrap(~type, labeller = labeller(type = c(
    "definition" = "Category Definition",
    "name" = "Category Name"
  ))) +
  coord_cartesian(ylim = c(0, 1.1)) +
  scale_y_continuous(breaks = seq(0, 1, by=0.25)) +
  scale_x_discrete(labels = c("Ucoop", "Udef")) +
  scale_fill_few("Dark", labels = c("Base", "Res1", "Res2", "Res3")) +
  plot_thm

plot_3 <- ggplot(subdf.unames.1, aes(y = treatment, x = count, color = treatment)) +
  geom_point(size = 3, position = position_jitter(width = 0), alpha = 0.6) +
  scale_colour_wsj("colors6") +
  coord_cartesian(xlim = c(0, 50)) + 
  labs(y = "Treatment", x = "Category Name Count (n = 50)", title = "Unique Category Name Counts (Tests 12 - 14)") +
  plot_thm +
  theme(axis.title.x = element_text(size = 18), legend.position = "none")

plot_4 <- ggplot(subdf.unames.2, aes(y = configs, x = count, color = configs)) +
  geom_point(size = 3, position = position_jitter(width = 0), alpha = 0.6) +
  scale_colour_few("Dark") +
  scale_y_discrete(labels = c("Base", "Res1", "Res2", "Res3")) +
  coord_cartesian(xlim = c(0, 50)) + 
  labs(y = "Model Configurations", x = "Category Name Count (n = 50)", title = "Unique Category Name Counts (Tests 14 - 17)") +
  plot_thm +
  theme(axis.title.x = element_text(size = 18), legend.position = "none")

plot_5 <- ggplot(subdf.sim.3, aes(x = instance, y = sim, fill = treatment)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.5) +
  geom_errorbar(aes(ymin = sim - sim_sd, ymax = sim + sim_sd), 
                width = 0.2, linewidth = 0.8, 
                position = position_dodge(width = 0.6)) +
  labs(y = "Cosine Similarity", fill = "Treatment",
       title = "Stage 1 Categories (Tests 21 - 23)") +
  facet_wrap(~type, labeller = labeller(type = c(
    "definition" = "Category Definition",
    "name" = "Category Name"
  ))) +
  scale_y_continuous(breaks = seq(0, 1, by=0.25)) +
  scale_x_discrete(labels = c("Ucoop", "Udef")) +
  scale_fill_wsj("colors6") +
  plot_thm

plot_6 <- ggplot(subdf.sim.4, aes(x = instance, y = sim, fill = configs)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.6), width = 0.5) +
  geom_errorbar(aes(ymin = sim - sim_sd, ymax = sim + sim_sd), 
                width = 0.2, linewidth = 0.8, 
                position = position_dodge(width = 0.6)) +
  labs(y = "Cosine Similarity", x = "Instance", fill = "",
       title = "Stage 1 Categories (Tests 23 - 26)") +
  facet_wrap(~type, labeller = labeller(type = c(
    "definition" = "Category Definition",
    "name" = "Category Name"
  ))) +
  coord_cartesian(ylim = c(0, 1.1)) +
  scale_y_continuous(breaks = seq(0, 1, by=0.25)) +
  scale_x_discrete(labels = c("Ucoop", "Udef")) +
  scale_fill_few("Dark", labels = c("Base", "Res1", "Res2", "Res3")) +
  plot_thm

plot_7 <- ggplot(subdf.unames.3, aes(y = treatment, x = count, color = treatment)) +
  geom_point(size = 3, position = position_jitter(width = 0), alpha = 0.6) +
  scale_colour_wsj("colors6") +
  coord_cartesian(xlim = c(0, 100)) + 
  labs(y = "Treatment", x = "Category Name Count (n = 100)", title = "Unique Category Name Counts (Tests 21 - 23)") +
  plot_thm +
  theme(axis.title.x = element_text(size = 18), legend.position = "none")

plot_8 <- ggplot(subdf.unames.4, aes(y = configs, x = count, color = configs)) +
  geom_point(size = 3, position = position_jitter(width = 0), alpha = 0.6) +
  scale_colour_few("Dark") +
  scale_y_discrete(labels = c("Base", "Res1", "Res2", "Res3")) +
  coord_cartesian(xlim = c(0, 100)) + 
  labs(y = "Model Configurations", x = "Category Name Count (n = 100)", title = "Unique Category Name Counts (Tests 23 - 26)") +
  plot_thm +
  theme(axis.title.x = element_text(size = 18), legend.position = "none")

ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t12-t14-stage_1_similarities.pdf"), plot_1, width = 12, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t14-t17-stage_1_similarities.pdf"), plot_2, width = 12, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t12-t14-stage_1_ucategories.pdf"), plot_3, width = 10, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t14-t17-stage_1_ucategories.pdf"), plot_4, width = 10, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t21-t23-stage_1_similarities.pdf"), plot_5, width = 12, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t23-t26-stage_1_similarities.pdf"), plot_6, width = 12, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t21-t23-stage_1_ucategories.pdf"), plot_7, width = 10, height = 6)
ggsave(paste0(Sys.getenv("PROJECT_DIRECTORY"), "documentation/plots/t23-t26-stage_1_ucategories.pdf"), plot_8, width = 10, height = 6)

