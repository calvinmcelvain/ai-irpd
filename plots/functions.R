# Packages
library(dplyr)
library(ggplot2)
library(patchwork)


# Summary statistics function
sum_stats <- function(data, group, stat) {
  sum_stats <- data %>%
    group_by(.data[[group]]) %>%
    summarise(
      means = mean(.data[[stat]], na.rm = T),
      vars = var(.data[[stat]], na.rm = T)
    )
  return(sum_stats)
}


# y-limit calculation function (between two plots or more)
calculate_y_limits <- function(..., x_var, y_var, group_var) {
  data_list <- list(...)
  limits <- lapply(data_list, function(df) {
    df_summary <- df %>%
      group_by(!!sym(x_var), !!sym(group_var)) %>%
      summarise(
        ymin = mean_cl_95(!!sym(y_var))$ymin,
        ymax = mean_cl_95(!!sym(y_var))$ymax,
        .groups = "drop"
      )
    range(df_summary$ymin, df_summary$ymax)
  })
  c(min(sapply(limits, `[`, 1)) - 0.01, max(sapply(limits, `[`, 2)) + 0.01)
}


# 95% CI functions
mean_cl_95 <- function(y) {
  mean_y <- mean(y)
  stderr_y <- sd(y) / sqrt(length(y) - 1)
  error_margin <- qt(0.975, length(y) - 1) * stderr_y
  data.frame(
    y = mean_y, 
    ymin = mean_y - error_margin, 
    ymax = mean_y + error_margin
  )
}

point_cl_95 <- function(y) {
  mean_y <- mean(y)
  return(mean_y)
}

ub_cl_95 <- function(y) {
  mean_y <- mean(y)
  stderr_y <- sd(y) / sqrt(length(y) - 1)
  error_margin <- qt(0.975, length(y) - 1) * stderr_y
  ub <- mean_y + error_margin
  return(ub)
}

lb_cl_95 <- function(y) {
  mean_y <- mean(y)
  stderr_y <- sd(y) / sqrt(length(y) - 1)
  error_margin <- qt(0.975, length(y) - 1) * stderr_y
  lb <- mean_y - error_margin
  if (lb < 0) {
    lb <- 0
  }
  return(lb)
}


# Double plot save function
combine_and_save <- function(plot1, plot2, filename, layout = "side_by_side") {
  if (layout == "side_by_side") {
    combined_plot <- plot1 + plot2 + plot_layout(ncol = 2)
    ggsave(filename, plot = combined_plot, width = 14, height = 6, dpi = 1000)
  } else if (layout == "stacked") {
    combined_plot <- plot1 + plot2 + plot_layout(ncol = 1)
    ggsave(filename, plot = combined_plot, width = 14, height = 8, dpi = 1000)
  }
  return(combined_plot)
}


# Quad plot save function
combine_and_save_quad <- function(plot1, plot2, plot3, plot4, filename) {
  combined_plot <- plot1 + plot2 + plot3 + plot4 +
    plot_layout(ncol = 2, nrow = 2)
  ggsave(filename, plot = combined_plot, width = 16, height = 12, dpi = 1000)
  return(combined_plot)
}


# Single plot save function
plot_save <- function(plot, filename) {
  ggsave(filename = filename, plot = plot, width = 12, height = 8, dpi = 1000)
}
