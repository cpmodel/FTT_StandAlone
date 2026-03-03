
# Load libraries ----------------------------------------------------------

library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(viridis)
library(vroom)
library(ggpattern)


# Set up config file & paths --------------------------------------------

script_dir <- this.path::this.dir()
wd <- dirname(dirname(dirname(script_dir)))
# set the working directory as FTT folder
setwd(wd)

# import config
config_path <- "Emulation/code/config/config.json"
config <- jsonlite::fromJSON(config_path) 

# main paths TODO: are any unnecessary
data_path <- "Emulation/data/"

# Prediction file names
out_in_preds <- "predictions/preds_IN_old.csv"
out_gbl_preds <- "predictions/preds_GBL_old.csv"

# Load emulator prediction data
df_gbl <- vroom(paste0(data_path, out_gbl_preds))
df_in <- vroom(paste0(data_path, out_in_preds))



# Policy factors function -------------------------------------------------

# add plotting vars to dataframe
add_policy_factors <- function(df){
  df %>% 
  # Create US policy factor variable
  mutate(
    IN_subsidy = factor(
      case_when(
        IN_price_pol == 0 ~ "Subsidy - None",
        IN_price_pol == 0.5 ~ "Subsidy - Mid",
        IN_price_pol == 1  ~ "Subsidy - High",
        TRUE                ~ NA_character_
      ),
      #levels = c("Subsidy - None", "Subsidy - Mid", "Subsidy - High")), # reversed for grid order
      levels = c("Subsidy - High", "Subsidy - Mid", "Subsidy - None")
    ), # reversed for grid order
    
    IN_cp = factor(
      case_when(
        IN_cp_pol == 1  ~ "CP - High",
        IN_cp_pol == 0.5 ~ "CP - Mid",
        IN_cp_pol == 0 ~ "CP - None"
      ),
      levels = c( "CP - High", "CP - Mid", "CP - None")
    ),
    # levels = c("CP - None", "CP - Mid", "CP - High")),
    
    IN_phase = factor(
      case_when(
        IN_phase_pol == 0 ~ "Phaseouts - None",
        IN_phase_pol == 0.5 ~ "Phaseouts - Mid",
        IN_phase_pol == 1  ~ "Phaseouts - High"
        
      ),
      levels = c("Phaseouts - None", "Phaseouts - Mid", "Phaseouts - High"),
      #levels = c("Phaseouts - High" , "Phaseouts - Mid", "Phaseouts - None")
    ),
    `Policy Combination` = factor (
      case_when(
      IN_phase_pol == 0 & IN_cp_pol == 0 & IN_price_pol == 0 ~ 'Baseline',
      IN_phase_pol == 0 & IN_cp_pol == 1 & IN_price_pol == 1 ~ 'Sub-CP',
      IN_phase_pol == 1 & IN_cp_pol == 1 & IN_price_pol == 0 ~ 'CP-Phaseout',
      IN_phase_pol == 1 & IN_cp_pol == 0 & IN_price_pol == 1 ~ 'Sub-Phaseout',
      IN_phase_pol == 1 & IN_cp_pol == 1 & IN_price_pol == 1 ~ 'CP-Sub-Phaseout'
      ),
      levels = c('Baseline', 'Sub-CP', 'CP-Phaseout', 'Sub-Phaseout', 'CP-Sub-Phaseout'),
  ),
    CN_phase = factor(
      case_when(
        CN_phase_pol == 0 ~ "Baseline",
        CN_phase_pol == 0.5 ~ "Mid-level Policy",
        CN_phase_pol == 1  ~ "High-level Policy"
        
      ),
      levels = c("High-level Policy", "Mid-level Policy", "Baseline"),
      #levels = c("High-lvel Policy", "Mid-level Policy", "Current Policy"),
    ),
    `Wind Learning` = factor(
      case_when(
        learning_wind < 0.3 ~ "Low",
        learning_wind >= 0.3 & learning_wind < 0.6 ~ "Medium",
        learning_wind >= 0.6 ~ "High"
      ),
      levels = c("Low", "Medium", "High")),
    `Solar Learning` = factor(
      case_when(
        learning_solar < 0.3 ~ "Low",
        learning_solar >= 0.3 & learning_solar < 0.6 ~ "Medium",
        learning_solar >= 0.6 ~ "High"
      ),
      levels = c("Low", "Medium", "High")
      
    ),
    lead_onsh = factor(
      case_when(
        lead_onshore < 0.34 ~ "Low onshore lead",
        lead_onshore >= 0.34 & lead_onshore < 0.67 ~ "Medium onshore lead",
        lead_onshore >= 0.67 ~ "High onshore lead"
      ),
      levels = c("Low onshore lead", "Medium onshore lead", "High onshore lead")
      #levels = c("High onshore lead", "Medium onshore lead", "Low onshore lead")
      
    ),
   demand = factor(
      case_when(
        elec_demand < 0.5 ~ "Low demand",
        # elec_demand >= 0.33 & elec_demand < 0.66 ~ "Mid demand",
        elec_demand >= 0.5 ~ "High demand"
      ),
      levels = c("Low demand", "High demand") # Mid demand
    ),
   lead_sol = factor(
     case_when(
       lead_solar < 0.5 ~ "Short",
       lead_solar >= 0.5 ~ "Long"
     ),
     levels = c("Short", "Long")
   ),
   leads = factor(
     case_when(
       lead_solar < 0.5 & lead_onshore < 0.5 ~ "Short tech lead",
       lead_solar >= 0.5 & lead_onshore >= 0.5 ~ "Long tech lead"
     ),
     levels = c("Short tech lead", "Long tech lead")
   ),
   # lead_total = factor(
   #   case_when(
   #     lead_solar < 0.33 & lead_onshore < 0.33 & lead_commission < 0.33 ~ "Fast lead times",
   #     lead_solar >= 0.33 & lead_solar < 0.66 &
   #       lead_onshore >= 0.33 & lead_onshore < 0.66 &
   #       lead_commission >= 0.33 & lead_commission < 0.66 ~ "Medium lead times",
   #     lead_solar >= 0.66 & lead_onshore >= 0.66 & lead_commission >= 0.66 ~ "Slow lead times"
   #   ),
   #   levels = c("Fast lead times", "Medium lead times", "Slow lead times")
     #levels = c("Slow rollout", "Medium rollout", "Fast rollout" )
   #),
     lead_total = factor(
       case_when(
         lead_solar < 0.5 & lead_onshore < 0.5 & lead_commission < 0.5 ~ "Fast lead times",
         lead_solar >= 0.5 & lead_onshore >= 0.5 & lead_commission >= 0.5 ~ "Slow lead times"
       ),
       #levels = c("Fast rollout", "Slow rollout")
       levels = c("Slow lead times", "Fast lead times")
   ),
  cr_total = factor(
    case_when(
      cr_wind < 0.5 & cr_solar < 0.5 ~ "Low Cannibalisation",
      cr_solar >= 0.5 & cr_wind >= 0.5 ~ "High Cannibalisation"
    ),
    levels = c("Low Cannibalisation", "High Cannibalisation")
  ),
   grid = factor(
     case_when(
       lead_commission < 0.5  ~ "Short commission",
       lead_commission >= 0.5 ~ "Long commission"
     ),
     levels = c("Short commission", "Long commission")
   ),
   # demand = factor(
   #   case_when(
   #     elec_demand < 0.33 ~ "Low Demand",
   #     elec_demand >= 0.5 ~ "High Demand"
   #   ),
   #   levels = c("Low Demand", "High Demand")
   # ),
    discount_rate = factor(
      case_when(
        discr >= 0.5 ~ 'High Discount Rate',
        # discr < 0.66 & discr >= 0.33 ~ 'Mid DR',
        discr < 0.5 ~ 'Low Discount Rate'
      ),
      #levels = c('High Discount Rate', 'Low Discount Rate')
      levels = c('Low Discount Rate', 'High Discount Rate')
      
    ),
    coal_p = factor(
      case_when(
        coal_price < 0.5 ~ "Low Coal Price",
        # coal_price >= 0.3 & coal_price < 0.6 ~ "Medium Price",
        coal_price >= 0.5 ~ "High Coal Price"
      ),
      levels = c("Low Coal Price", "High Coal Price")
    ),
    
    tech_p = factor(
      case_when(
        tech_potential < 0.3 ~ "Low",
        tech_potential >= 0.3 & tech_potential < 0.6 ~ "Medium",
        tech_potential >= 0.6 ~ "High"
      ),
      levels = c("Low", "Medium", "High")
    )
  )
}



# Global plots ------------------------------------------------------------

# Add in policy factors
df_gbl <- add_policy_factors(df_gbl)


# Emissions GLOBAL --------------------------------------------------------


# Statistics for plot
summary_stats_3 <- df_gbl %>%
  subset(emulator %in% c("MEWE_GBL_2030", "MEWE_GBL_2050")) %>%
  filter(!is.na(cr_total) & !is.na(lead_total)) %>%
  filter(CN_phase != 'High-level Policy') %>%
  
  # Create US policy factor variable
  mutate(
    year = factor(
      case_when(
        emulator == "MEWE_GBL_2030" ~ "2030",
        emulator == "MEWE_GBL_2050" ~ "2050"
      ),
      levels = c("2030",  "2050"))) %>%
  group_by(CN_phase, lead_total, cr_total, emulator) %>%
  summarise(median_pred = median(prediction/1000, na.rm = TRUE),
            p05_pred = quantile(prediction/1000, 0.05, na.rm = TRUE),
            p95_pred = quantile(prediction/1000, 0.95, na.rm = TRUE),
            .groups = 'drop') %>%
  # Change negative values to 0 for plotting
  # Negative values come from the modeled increase in negative emissions technologies
  mutate(
    median_pred = pmax(median_pred, 0))



# Plot
p <- df_gbl %>%
  subset(emulator %in% c("MEWE_GBL_2030",'MEWE_GBL_2050')) %>%
  filter(!is.na(cr_total) & !is.na(lead_total)) %>%
  filter(CN_phase != 'High-level Policy') %>%
  mutate(
    year = factor(case_when(
      emulator == "MEWE_GBL_2030" ~ "2030",
      emulator == "MEWE_GBL_2050" ~ "2050"
    ), levels = c("2030", "2050")),
    prediction = prediction/1000
  ) %>%
  ggplot(aes(x = prediction, fill = year)) +
  
  # # Fixed geom_vline calls - NO extra commas!
  # geom_vline(data = summary_stats_3, aes(xintercept = p05_pred),
  #            color = "black", linetype = "dotted", size = 0.8, show.legend = FALSE) +
  # geom_vline(data = summary_stats_3, aes(xintercept = p95_pred), 
  #            color = "black", linetype = "dotted", size = 0.8, show.legend = FALSE) +
  # 
  # Fixed geom_density - remove conflicting fill=year
  geom_density(alpha = 0.5, adjust = 3) + 
  
  geom_vline(data = summary_stats_3, aes(xintercept = median_pred, color = emulator),
             linetype = "dashed", size = 0.6, show.legend = FALSE) +
  
  facet_grid(CN_phase ~ lead_total + cr_total, labeller = label_value, scales = "fixed") +
  labs(x = expression("Power Sector Emissions Levels (GtCO"[2]*"/year)"), 
       y = "Density", fill = "Year") + 
  scale_fill_viridis_d(option = "C") +
  scale_x_continuous(limits = c(0, 28)) +
  scale_color_manual(values = c("MEWE_GBL_2030" = "purple", "MEWE_GBL_2050" = "black")) +
  theme(
    legend.position = "bottom",
    axis.title.x  = element_text(size = 14, face = "bold"),
    axis.title.y  = element_text(size = 14, face = "bold"),
    axis.text.x   = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y   = element_text(size = 12),
    strip.text    = element_text(size = 13),
    legend.title  = element_text(size = 14, face = "bold"),
    legend.text   = element_text(size = 12),
    plot.title    = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.spacing.x = unit(0.5, "lines")
  )

# Save to figures folder - FIG 4
ggsave(paste0(data_path, "figures/emiss_gbl_fig4.png"), p, 
       width = 10, height = 8, dpi = 100)


# delete df for RAM
df_gbl <- NULL


# India plots -------------------------------------------------------------

# Add vars for plotting
df_in <- add_policy_factors(df_in)


# Capacity INDIA ----------------------------------------------------------

# compute statistics for plot
summary_stats <- df_in %>% 
  filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030") & 
           IN_cp == "CP - None") %>%
  group_by(year, id, sample_id) %>% 
  mutate(total_value = sum(prediction, na.rm = TRUE)) %>% 
  ungroup() %>%
  group_by(IN_subsidy, IN_phase, emulator) %>%
  summarise(median_pred = median(prediction, na.rm = TRUE),
            p25_pred = quantile(prediction, 0.05, na.rm = TRUE),
            p75_pred = quantile(prediction, 0.95, na.rm = TRUE),
            .groups = 'drop') %>%
  bind_rows(
    df_in %>%
      filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030")) %>% #& 
              # IN_cp == "CP - None") %>%
      group_by(year, id, sample_id) %>% 
      mutate(total_value = sum(prediction, na.rm = TRUE)) %>% 
      ungroup() %>%
      group_by(IN_subsidy, IN_phase) %>%
      summarise(median_total = median(total_value, na.rm = TRUE),
                p25_total = quantile(total_value, 0.05, na.rm = TRUE),
                p75_total = quantile(total_value, 0.95, na.rm = TRUE),
                .groups = 'drop') %>%
      mutate(emulator = "Total")
  )

# Compute total_value per sample
total_df <- df_in %>%
  filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030"),
         IN_cp == "CP - None") %>%
  mutate(
    curve_type = dplyr::recode(
      emulator,
      "MEWK_solar_IN_2030"   = "Solar PV",
      "MEWK_onshore_IN_2030" = "Onshore"
    )
  ) %>%
  group_by(year, id, sample_id, IN_subsidy, IN_phase) %>%
  summarise(total_value = sum(prediction, na.rm = TRUE),
            curve_type = "Total Capacity", .groups = "drop") 

# Compute total density for each facet and extract y value at x = 393
total_density <- total_df %>%
  group_by(IN_subsidy, IN_phase) %>%
  summarise(dens = list(density(total_value, na.rm = TRUE)), .groups = "drop") %>%
  mutate(dens_df = purrr::map(dens, ~ data.frame(x = .x$x, y = .x$y))) %>%
  tidyr::unnest(dens_df) 

# create threshold for target and label
threshold <- 393
total_density$region <- ifelse(total_density$x >= threshold, "above", "below")

# simplify labelling
tech_df_plot <- df_in %>% 
  mutate(
    curve_type = dplyr::recode(
      emulator,
      "MEWK_solar_IN_2030"   = "Solar PV",
      "MEWK_onshore_IN_2030" = "Onshore"
    )
  ) %>%
  filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030"))
         
# cut of extreme values for plotting (explained in paper)
total_density_plot <- total_density |>
  filter(x >= 0, x <= 800) |>
  mutate(
    curve_type = case_when(
      region == "below" ~ "Total (below target)",
      region == "above" ~ "Total (above target)"
    )
  )

# plot
p <- ggplot(tech_df_plot) +
  geom_density(
    aes(x = prediction, fill = curve_type),
    alpha = 0.5, adjust = 2
  ) +
  facet_grid(IN_subsidy ~ IN_phase, labeller = label_value) +
  
  # Solid grey for below
  geom_area(
    data = subset(total_density_plot, region == "below"),
    aes(x, y, fill = curve_type),
    stat = "identity",
    colour = NA,
    alpha = 0.8
  ) +
  # Hatched for above
  geom_area_pattern(
    data = subset(total_density_plot, region == "above"),
    aes(x, y, fill = curve_type, pattern = region),
    stat = "identity",
    colour = NA,
    alpha = 0.2,
    pattern = "stripe",
    pattern_fill = "black",
    pattern_density = 0.4,
    pattern_spacing = 0.03,
    pattern_angle = 45
  ) +
  
  # Separate lines if needed, or drop show.legend = FALSE on one
  geom_line(
    data = total_density_plot,
    aes(x, y, colour = curve_type),
    linewidth = 0.7
  ) +
  
  # Vlines unchanged
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = median_total),
             color = "black", linetype = "solid", size = 0.6, show.legend = FALSE) +
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = p25_total),
             color = "black", linetype = "dotted", size = 0.8, show.legend = FALSE) +
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = p75_total),
             color = "black", linetype = "dotted", size = 0.8, show.legend = FALSE) +
  
  scale_fill_manual(
    values = c(
      "Onshore" = "#0072B2",
      "Solar PV" = "#D55E00",
      "Total (below target)" = "grey80",
      "Total (above target)" = "grey80"
    )
  ) +
  scale_colour_manual(
    values = c(
      "Total (below target)" = "black",
      "Total (above target)" = "black"
    ),
    guide = "none"
  ) +
  scale_pattern_manual(
    values = c("below" = "none", "above" = "stripe"),
    guide = guide_legend(
      title = "Total Capacity Region",
      override.aes = list(fill = "grey80", pattern_fill = "black")
    )
  ) +
  
  
  labs(
    x = "Capacity (GW)", 
    y = "Density",
    fill = "Technology",
    pattern = ""
  ) +
  
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12),
    strip.text = element_text(size = 13),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.spacing.x = unit(0.5, "lines")
  )

# save 
ggsave(paste0(data_path, "figures/capacity_in_fig5.png"), p, 
       width = 10, height = 8, dpi = 100)



# Emission and demand INDIA -----------------------------------------------

# Reverse levels for intuitive plot
df_in$IN_phase <- factor(df_in$IN_phase, levels = rev(levels(df_in$IN_phase)))

# compute summary stats for plot
summary_stats_3 <- df_in %>% subset(emulator %in% c( 'MEWE_IN_2050') & #'MEWE_IN_2030',
                                   IN_cp == 'CP - None' & 
                                   IN_subsidy == 'Subsidy - None' & 
                                   IN_phase != 'Phaseouts - High' &
                                   !is.na(cr_total)) %>%
  
  # Create US policy factor variable
  mutate(
    year = factor(
      case_when(
        #emulator == "MEWE_IN_2030" ~ "2030",
        emulator == "MEWE_IN_2050" ~ "2050"
      ),
      levels = c(  "2050")),
    demand = factor(
      case_when(
        demand == "Low demand"  ~ "Low",
        demand == "High demand" ~ "High"
      ),
      levels = c("Low", "High")
    )) %>% #"2030",
  group_by(IN_phase, demand, cr_total, emulator, coal_p) %>%
  summarise(median_pred = median(prediction/1000, na.rm = TRUE),
            .groups = 'drop') 

# plot
p <- df_in %>% 
  subset(
    emulator %in% c( 'MEWE_IN_2050') & #'MEWE_IN_2030',
      IN_cp == 'CP - None' & 
      IN_subsidy == 'Subsidy - None' & 
      IN_phase != 'Phaseouts - High' &
      !is.na(cr_total)
  ) %>%
  
  mutate(
    year = factor(
      case_when(
        #emulator == "MEWE_IN_2030" ~ "2030",
        emulator == "MEWE_IN_2050" ~ "2050"
      ),
      levels = c( "2050")),#"2030",
    demand = factor(
      case_when(
        demand == "Low demand" ~ "Low",
        demand == "High demand" ~ "High"
        ),
        levels = c("Low", "High")),
    
  ) %>%
  
  ggplot(aes(x = prediction/1000)) +

  geom_density(
    aes(
      fill  = demand,
      color = 'black'
    ),
    alpha = 0.9,
    adjust = 2, 
    size = 0.5, 
    linetype = 'solid'
  ) +
  geom_vline(
    data = summary_stats_3, 
    aes(xintercept = median_pred, colour = demand),
    linetype = "dashed", size = 0.8, show.legend = FALSE
  ) +

  facet_grid(IN_phase ~ cr_total + coal_p, labeller = label_value, scales = 'fixed') +
  
  labs(
    x = expression("Power Sector Emissions Levels (GtCO"[2]*"/year)"), 
    y = "Density",
    fill = "Electricity Demand",
  ) +
  
  scale_fill_viridis_d(option = "C") +
  scale_color_viridis_d(option = "C") +
  scale_x_continuous(limits = c(0, 5.5)) + 
  
  scale_fill_manual(
    values = c(
      # "2030 - Low"  = "#236192",
      # "2030 - High" = "#3182bd",
      "Low"  = "#e34a33",
      "High" = "#fc8d59"
    )
  ) +
  scale_color_manual(
    values = c(
      "Low"  = "#e34a33",
      "High" = "#fc8d59"
    )
  ) +
  
  guides(color = "none") +
  theme(
    legend.position = "bottom",
    # Axis titles
    axis.title.x  = element_text(size = 14, face = "bold"),
    axis.title.y  = element_text(size = 14, face = "bold"),
    # Axis tick labels
    axis.text.x   = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y   = element_text(size = 12),
    # Facet strip labels
    strip.text    = element_text(size = 13),
    # Legend title and items
    legend.title  = element_text(size = 14, face = "bold"),
    legend.text   = element_text(size = 12),
    # Plot title
    plot.title    = element_text(size = 18, face = "bold", hjust = 0.5),
    # Adjust spacing if needed
    panel.spacing.x = unit(0.5, "lines")
)

ggsave(paste0(data_path, "figures/emiss_in_fig6.png"), p, 
       width = 10, height = 8, dpi = 100)



# Robustness INDIA --------------------------------------------------------

## Create multiple summary tables for each output of interest

# Capacity - solar & onshore 2030
threshold <- 393
summ_1 <- df_in %>%
  filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030")) %>%
  group_by(year, id, sample_id) %>%
  mutate(total_value = sum(prediction, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(`Policy Combination`, lead_total, cr_total, demand, discount_rate) %>%
  summarise(
    total_obs = n()/2, # divide by 2 to correct for double count
    breach_threshold = sum(total_value > threshold, na.rm = TRUE)/2,
    target = "Capacity",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(total_value)) %>%
  drop_na()

# elec price of generation (weighted average)
threshold <- 68 # baseline projection for 2030
summ_2 <- df_in %>% 
  filter(emulator %in% c("MEWP_elec_price_IN_2030")) %>%
  group_by(`Policy Combination`, lead_total, cr_total, demand, discount_rate) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction < threshold, na.rm = TRUE),
    target = "Electricity Price",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()


#  emissions (yearly 2030)
threshold <- 1000
summ_3 <- df_in %>% 
  filter(emulator %in% c("MEWE_IN_2030")) %>%
  group_by(`Policy Combination`, lead_total, cr_total, demand, discount_rate) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction < threshold, na.rm = TRUE),
    target = "Emissions",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()

## shares of generation mix
threshold <- 0.55
summ_4 <- df_in %>% 
  filter(emulator %in% c("MEWS_renew_IN_2030")) %>%
  group_by(`Policy Combination`, lead_total, cr_total, demand, discount_rate) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction > threshold, na.rm = TRUE),
    target = "Shares",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()

# Combine sunmmary tables and add ordering col
summ <- rbind(summ_2, summ_3, summ_4) %>% 
  group_by(`Policy Combination`, lead_total, cr_total, demand, discount_rate) %>%
  mutate(combined_prop = round(sum(`Proportion achieving target`), 2))

# edit for plotting
plot_df <- summ %>% 
  bind_rows(summ) %>%
  mutate(
    target = factor(target,
                    levels = c("Electricity Price", "Emissions", "Shares"), 
                    labels = c("Electricity Price","Emissions", "Shares"))
  ) %>%
  ungroup() 

# CB friendly pal
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



# Final plot for proportions over targets
p <- ggplot(plot_df,
       aes(
         x    = `Policy Combination`,
         y    = `Proportion achieving target`,
         fill = target
       )) +
  geom_col(position = position_dodge(0.7), width = 0.7) +
  facet_grid(
    cr_total + discount_rate ~ lead_total + demand,
    labeller = label_value,
    scales = "free_x"        # free_x so each facet has its own ordering
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    name    = "Proportion achieving target",
    breaks    = seq(0, 1, by = 0.25)) +
  
  scale_fill_manual(
    values = c(
      "Electricity Price" = "#21918c",
      "Emissions"         = "#443983",
      "Shares"           = "#000000"
    )
  ) +
  
  
  labs(
    x    = "Policy Combination",
    y    = "Proportion achieving target",
    fill = "Target type"
  ) +
  
  theme(
    legend.position = "bottom",
    # Axis titles
    axis.title.x  = element_text(size = 14, face = "bold"),
    axis.title.y  = element_text(size = 14, face = "bold"),
    # Axis tick labels
    axis.text.x   = element_text(size = 12, face = "bold", angle = 45, hjust = 1),
    axis.text.y   = element_text(size = 12),
    # Facet strip labels
    strip.text    = element_text(size = 13),
    # Legend title and items
    legend.title  = element_text(size = 14, face = "bold"),
    legend.text   = element_text(size = 12),
    # Plot title
    plot.title    = element_text(size = 18, face = "bold", hjust = 0.5),
    # Adjust spacing if needed
    panel.spacing.x = unit(0.5, "lines")
  )

# save
ggsave(paste0(data_path, "figures/robustness_in_fig7.png"), p, 
       width = 10, height = 8, dpi = 100)




















