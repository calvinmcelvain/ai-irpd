# Packages
library(dotenv)
library(ggplot2)
library(readxl)
library(wesanderson)
library(forcats)

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
stage_1_name <- read_xlsx(data_path, sheet = "stage_1_name")
stage_1_def <- read_xlsx(data_path, sheet = "stage_1_def")
stage_1_cats <- read_xlsx(data_path, sheet = "stage_1_cats")

## Base subsets
stage_1_name_base <- subset(stage_1_name, config == "base")
stage_1_def_base <- subset(stage_1_def, config == "base")
stage_1_cats_base <- subset(stage_1_cats, config == "base")

## Models subsets
stage_1_name_models <- subset(stage_1_name, treatment == "Merged")
stage_1_def_models <- subset(stage_1_def, treatment == "Merged")
stage_1_cats_models <- subset(stage_1_cats, treatment == "Merged")

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plots

error_plot <- function(
  data, x_var, y_var, color_var, sets, x_axis_title, title_text, pal, y_limits
) {
  p <- ggplot(data, aes(x = !!sym(x_var),
                        y = !!sym(y_var),
                        color = !!sym(color_var))) +
    stat_summary(
      fun.data = mean_cl_95,
      geom = "errorbar",
      width = 0.2,
      aes(group = !!sym(color_var)),
      position = position_dodge(width = 0.6)
    ) +
    stat_summary(
      fun = mean,
      geom = "point",
      size = 3,
      position = position_dodge(width = 0.6)
    ) +
    stat_summary(
      fun = mean,
      geom = "text",
      aes(label = round(after_stat(y), 2), group = !!sym(color_var)),
      position = position_dodge(width = 0.6),
      hjust = 1.4,
      color = "black",
      size = 4
    ) +
    labs(
      title = title_text,
      x = x_axis_title,
      y = "Cosine Similarity",
      color = "Instance Type"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.line = element_line(color = "black"),
      axis.text.x = element_text(color = "black", size = 13),
      legend.position = "bottom",
      legend.text = element_text(size = 12),
      strip.text = element_text(size = 15, face = "bold")
    ) +
    scale_color_manual(
      values = pal,
      labels = c(
        "ucoop" = "Unilateral Cooperation",
        "udef" = "Unilateral Defection"
      )
    ) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.05)) +
    coord_cartesian(ylim = y_limits)
  if (sets) {
    p <- p +
      aes(shape = set) +
      facet_wrap(
        ~set,
        labeller = labeller(
          set = c("Set_1" = "Set 1 (N = 30)", "Set_2" = "Set 2 (N = 50)")
        )
      ) +
      scale_shape_manual(
        values = c("Set_1" = 16, "Set_2" = 17),
        labels = c("Set_1" = "Set 1", "Set_2" = "Set 2"),
        guide = "none"
      )
  }

  return(p)
}


bar_chart <- function(
  data, x_var, y_var, color_var, x_axis_title, title_text, legend_text, pal
) {
  data[[x_var]] <- fct_reorder(factor(data[[x_var]]), data[[y_var]], 
                               .desc = TRUE)
  ggplot(data, aes(x = !!sym(x_var), y = !!sym(y_var),
                   fill = !!sym(color_var))) +
    geom_bar(stat = "identity") +
    labs(
         title = title_text,
         x = x_axis_title,
         y = element_blank(),
         color = legend_text) +
    theme_minimal(base_size = 14) +
    theme(
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "bottom",
          legend.text = element_text(size = 12)) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_fill_manual(values = pal) +
    scale_shape_manual(values = c(16, 17, 18, 19))
}

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plotting and Saving

## Name and Definition similarity plot seperated by sets
base_limits <- calculate_y_limits(
  stage_1_name_base,
  stage_1_def_base,
  x_var = "treatment",
  y_var = "sim",
  group_var = "type"
)
models_limits <- calculate_y_limits(
  stage_1_name_models,
  stage_1_def_models,
  x_var = "config",
  y_var = "sim",
  group_var = "type"
)

plot1 <- error_plot(
  data = stage_1_name_base,
  x_var = "treatment",
  y_var = "sim",
  color_var = "type",
  sets = TRUE,
  x_axis_title = "Treatment",
  title_text = "Category Name Similarity",
  pal = wes_palettes$Cavalcanti1,
  y_limits = base_limits
)
plot2 <- error_plot(
  data = stage_1_def_base,
  x_var = "treatment",
  y_var = "sim",
  color_var = "type",
  sets = TRUE,
  x_axis_title = "Treatment",
  title_text = "Category Definition Similarity",
  pal = wes_palettes$Moonrise2,
  y_limits = base_limits
)
plot3 <- error_plot(
  data = stage_1_name_models,
  x_var = "config",
  y_var = "sim",
  color_var = "type",
  sets = TRUE,
  x_axis_title = "Model Configuration",
  title_text = "Category Name Similarity",
  pal = wes_palettes$Cavalcanti1,
  y_limits = models_limits
)
plot4 <- error_plot(
  data = stage_1_def_models,
  x_var = "config",
  y_var = "sim",
  color_var = "type",
  sets = TRUE,
  x_axis_title = "Model Configurations",
  title_text = "Category Definition Similarity",
  pal = wes_palettes$Moonrise2,
  y_limits = models_limits
)

combine_and_save(
  plot1 = plot1,
  plot2 = plot2,
  filename = file.path(output_dir, "stage_1_base_sets.png")
)
combine_and_save(
  plot1 = plot3,
  plot2 = plot4,
  filename = file.path(output_dir, "stage_1_models_sets.png")
)
combine_and_save_quad(
  plot1 = plot1,
  plot2 = plot1,
  plot3 = plot3,
  plot4 = plot4,
  filename = file.path(output_dir, "stage_1_sets.png")
)

## Name and Definition similarity plot combined sets
plot1 <- error_plot(
  data = stage_1_name_base,
  x_var = "treatment",
  y_var = "sim",
  color_var = "type",
  sets = FALSE,
  x_axis_title = "Treatment",
  title_text = "Category Name Similarity",
  pal = wes_palettes$Cavalcanti1,
  y_limits = base_limits
)
plot2 <- error_plot(
  data = stage_1_def_base,
  x_var = "treatment",
  y_var = "sim",
  color_var = "type",
  sets = FALSE,
  x_axis_title = "Treatment",
  title_text = "Category Definition Similarity",
  pal = wes_palettes$Moonrise2,
  y_limits = base_limits
)
plot3 <- error_plot(
  data = stage_1_name_models,
  x_var = "config",
  y_var = "sim",
  color_var = "type",
  sets = FALSE,
  x_axis_title = "Model Configuration",
  title_text = "Category Name Similarity",
  pal = wes_palettes$Cavalcanti1,
  y_limits = models_limits
)
plot4 <- error_plot(
  data = stage_1_def_models,
  x_var = "config",
  y_var = "sim",
  color_var = "type",
  sets = FALSE,
  x_axis_title = "Model Configurations",
  title_text = "Category Definition Similarity",
  pal = wes_palettes$Moonrise2,
  y_limits = models_limits
)

combine_and_save(
  plot1 = plot1,
  plot2 = plot2,
  filename = file.path(output_dir, "stage_1_base.png")
)
combine_and_save(
  plot1 = plot3,
  plot2 = plot4,
  filename = file.path(output_dir, "stage_1_models.png")
)
combine_and_save_quad(
  plot1 = plot1,
  plot2 = plot1,
  plot3 = plot3,
  plot4 = plot4,
  filename = file.path(output_dir, "stage_1.png")
)

## Category bar charts
plot1 <- bar_chart(
  data = subset(stage_1_cats_base, set == "Set_1"),
  x_var = "category",
  y_var = "count",
  color_var = "treatment",
  x_axis_title = "Category",
  title_text = "Between Treatment Categories (Base Model)",
  legend_text = "Treatment",
  pal = wes_palettes$Darjeeling1
)
plot2 <- bar_chart(
  data = subset(stage_1_cats_models, set == "Set_1"),
  x_var = "category",
  y_var = "count",
  color_var = "config",
  x_axis_title = "Category",
  title_text = "Between Model Configuration Categories (Merged Treatment)",
  legend_text = "Model Configuration",
  pal = wes_palettes$Darjeeling2
)
plot3 <- bar_chart(
  data = subset(stage_1_cats_base, set == "Set_2"),
  x_var = "category",
  y_var = "count",
  color_var = "treatment",
  x_axis_title = "Category",
  title_text = "Between Treatment Categories (Base Model)",
  legend_text = "Treatment",
  pal = wes_palettes$Darjeeling1
)
plot4 <- bar_chart(
  data = subset(stage_1_cats_models, set == "Set_2"),
  x_var = "category",
  y_var = "count",
  color_var = "config",
  x_axis_title = "Category",
  title_text = "Between Model Configuration Categories (Merged Treatment)",
  legend_text = "Model Configuration",
  pal = wes_palettes$Darjeeling2
)

plot_save(
  plot = plot2,
  filename = file.path(output_dir, "stage_1_categories_set1.png")
)
plot_save(
  plot = plot4,
  filename = file.path(output_dir, "stage_1_categories_set2.png")
)
