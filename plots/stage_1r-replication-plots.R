# Packages
library(dotenv)
library(ggplot2)
library(readxl)
library(wesanderson)
library(dplyr)

# Importing functions file
source(file.path(getwd(), "utils", "plots", "functions.R"))

# Getting data and output directory
config_dir <- file.path(getwd(), "configs", "configs.env")
dotenv::load_dot_env(config_dir)
data_dir <- paste0(Sys.getenv("PROJECT_DIRECTORY"), "data/output")

output_dir <- file.path(dirname(dirname(getwd())),
                        "OneDrive",
                        "ra_projects",
                        "irpd_coding",
                        "AI",
                        "replication_analysis",
                        "plots")


#-------------------------------------------------------------------------------
# Data import

## Importing data
data_path <- file.path(data_dir, "replication_data.xlsx")
stage_1r_v1 <- read_xlsx(data_path, sheet = "stage_1r_v1")
stage_1r_v2 <- read_xlsx(data_path, sheet = "stage_1r_v2")

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plots

bar_chart <- function(
  data, x_var, y_var, color_var, x_axis_title, title_text, pal
) {
  ggplot(data, aes(x = !!sym(x_var), y = !!sym(y_var),
                   fill = !!sym(color_var))) +
    geom_bar(stat = "identity") +
    labs(
         title = title_text,
         x = x_axis_title,
         y = "Keep Count",
         color = "Treatment") +
    theme_minimal(base_size = 14) +
    theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
          panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.text = element_text(size = 14),
          legend.title = element_blank()) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_fill_manual(values = pal) +
    scale_shape_manual(values = c(16, 17, 18, 19))
}

bar_chart2 <- function(
  data, x_var, y_var, color_var, x_axis_title, title_text, pal
) {
  ggplot(data, aes(x = !!sym(x_var), y = !!sym(y_var),
                   fill = !!sym(color_var))) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
         title = title_text,
         x = x_axis_title,
         y = "Number Categories Merged",
         color = "Instance Type") +
    theme_minimal(base_size = 14) +
    theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
          panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.text = element_text(size = 14),
          legend.title = element_blank()) +
    scale_fill_manual(
                      values = pal,
                      labels = c(
                        "ucoop" = "Unilateral Cooperation",
                        "udef" = "Unilateral Defection"
                      )) +
    scale_shape_manual(values = c(16, 17, 18, 19))
}

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plotting and saving

plot1 <- bar_chart(
  data = stage_1r_v1,
  x_var = "category",
  y_var = "keep_decision",
  color_var = "treatment",
  x_axis_title = "Categories",
  title_text = "Category Keep Decision",
  pal = wes_palettes$Zissou1
)

plot_save(
  plot = plot1,
  filename = file.path(output_dir, "stage_1r_v1.png")
)


plot1 <- bar_chart2(
  data = stage_1r_v2,
  x_var = "category",
  y_var = "merged_count",
  color_var = "type",
  x_axis_title = "Categories",
  title_text = "Categories Merged",
  pal = wes_palettes$Cavalcanti1
)

plot_save(
  plot = plot1,
  filename = file.path(output_dir, "stage_1r_v2.png")
)
