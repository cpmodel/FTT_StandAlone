
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(viridis)
library(vroom)
### Plotting

# Upload file or object
df <- vroom("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/IN_polcomp_lead_grid_2.csv")
#df <- vroom("C:/Users/ib400/Github/FTT_StandAlone/Emulation/data/predictions/GBL_emissions_amb.csv")


# adding in factor variables
df <- df %>% 
  # Create US policy factor variable
  mutate(
    IN_subsidy = factor(
      case_when(
        IN_price_pol == 0 ~ "Subsidy - None",
        IN_price_pol == 0.5 ~ "Subsidy - Mid",
        IN_price_pol == 1  ~ "Subsidy - High"
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
      #levels = c("Phaseouts - None", "Phaseouts - Mid", "Phaseouts - High"),
      levels = c("Phaseouts - High" , "Phaseouts - Mid", "Phaseouts - None")
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


      


####################################################################

# 2030 capacity and targert

####################################################################

# compute statistics for plot
summary_stats <- df %>% 
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
    df %>%
      filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030") & 
               IN_cp == "CP - None") %>%
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
total_df <- df %>%
  filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030"),
         IN_cp == "CP - None") %>%
  group_by(year, id, sample_id, IN_subsidy, IN_phase) %>%
  summarise(total_value = sum(prediction, na.rm = TRUE), .groups = "drop")

# line data
# Compute total density for each facet and extract y value at x = 393
total_density <- total_df %>%
  group_by(IN_subsidy, IN_phase) %>%
  summarise(dens = list(density(total_value, na.rm = TRUE)), .groups = "drop") %>%
  mutate(dens_df = purrr::map(dens, ~ data.frame(x = .x$x, y = .x$y))) %>%
  tidyr::unnest(dens_df) %>%
  group_by(IN_subsidy, IN_phase) %>%
  summarise(y_at_393 = approx(x, y, xout = 393)$y, .groups = "drop") %>%
  mutate(x = 393, xend = 393, yend = 0)

# Group by facet variables
facet_groups <- total_df %>%
  group_split(IN_subsidy, IN_phase)

# Get facet combinations
facet_keys <- total_df %>%
  distinct(IN_subsidy, IN_phase)

# Compute density per group and filter x >= 750
# shaded_df <- map2_dfr(facet_groups, seq_len(nrow(facet_keys)), function(group, i) {
#   dens <- density(group$total_value, na.rm = TRUE)
#   data.frame(
#     x = dens$x,
#     y = dens$y,
#     IN_subsidy = facet_keys$IN_subsidy[i],
#     IN_phase = facet_keys$IN_phase[i]
#   ) %>% filter(x >= 393)
# })
# shaded_df <- total_df %>%
#   group_split(IN_subsidy, IN_phase) %>%
#   map2_dfr(seq_len(nrow(facet_keys)), function(group, i) {
#     dens <- density(group$total_value, na.rm = TRUE)
#     tibble(
#       x = dens$x,
#       y = dens$y,
#       IN_subsidy = facet_keys$IN_subsidy[i],
#       IN_phase = facet_keys$IN_phase[i]
#     ) %>% filter(x >= 393)
#   })

shaded_df <- map2_dfr(facet_groups, seq_len(nrow(facet_keys)), function(group, i) {
  dens <- density(group$total_value, na.rm = TRUE)
  df <- data.frame(
    x = dens$x,
    y = dens$y,
    IN_subsidy = facet_keys$IN_subsidy[i],
    IN_phase = facet_keys$IN_phase[i]
  ) %>% filter(x >= 393)
  
  # close polygon for geom_area_pattern
  rbind(df, data.frame(x = max(df$x), y = 0,
                       IN_subsidy = facet_keys$IN_subsidy[i],
                       IN_phase = facet_keys$IN_phase[i]),
        data.frame(x = min(df$x), y = 0,
                   IN_subsidy = facet_keys$IN_subsidy[i],
                   IN_phase = facet_keys$IN_phase[i]))
})

# Invert levels to align with facets
levels(shaded_df$IN_subsidy) = c("Subsidy - None", "Subsidy - Mid", "Subsidy - High")
levels(shaded_df$IN_phase) = c("Phaseouts - None", "Phaseouts - Mid", "Phaseouts - High")


# Capacity distributions with percentile lines
ggplot(
  df %>% 
    filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030") &
             IN_cp == 'CP - None') %>%
    group_by(year, id, sample_id) %>% 
    mutate(total_value = sum(prediction, na.rm = TRUE)) %>%
    ungroup(),
  aes(x = prediction, fill = factor(emulator))
) +
  geom_density(alpha = 0.5, show.legend = FALSE, adjust = 2) +
  # Shaded area above threshold
  geom_area(data = shaded_df, aes(x = x, y = y),
             fill = "red", alpha = 0.3, inherit.aes = FALSE) +
  # geom_area_pattern(
  #   data = shaded_df,
  #   aes(x = x, y = y),
  #   pattern = "stripe",          # or "crosshatch", "circle", etc.
  #   pattern_fill = "red",        # color of the stripes
  #   pattern_density = 0.4,       # spacing of stripes
  #   pattern_angle = 45,          # angle of the stripes
  #   fill = NA,                   # no solid fill
  #   alpha = 0.6,                 # transparency of the pattern
  #   inherit.aes = FALSE
  # ) +
  # geom_vline(aes(xintercept = 393), color = "red") +
  geom_segment(
    data = total_density,
    aes(x = x, xend = xend, y = yend, yend = y_at_393),
    color = "red", size = 0.5, inherit.aes = FALSE
  ) +
  # geom_vline(
  #   data = redline_df,
  #   aes(xintercept = xint),
  #   color = "red", size = 0.7
  # ) +
  # Add dashed density line for total
  geom_density(aes(x = total_value, fill = "Total Capacity"), 
               linetype = "solid", size = 0.8, alpha = 0.4, show.legend = TRUE,
               adjust =  2) +

  # Facet
  facet_grid(IN_subsidy ~ IN_phase, labeller = label_value) +

  # # # Add median lines for each emulator
  # geom_vline(data = summary_stats %>% filter(emulator != "Total"),
  #            aes(xintercept = median_pred, color = emulator), show.legend = FALSE,
  #            linetype = "solid", size = 0.6,
  #            color = "black") +
  # # 
  # # # Add 25th and 75th percentile lines for each emulator
  # geom_vline(data = summary_stats %>% filter(emulator != "Total"),
  #            aes(xintercept = p25_pred, color = emulator), show.legend = FALSE,
  #            linetype = "dotted", size = 0.8,
  #            color = "black") +
  # geom_vline(data = summary_stats %>% filter(emulator != "Total"),
  #            aes(xintercept = p75_pred, color = emulator), show.legend = FALSE,
  #            linetype = "dotted", size = 0.8,
  #            color = "black") +

  # Add median and percentile lines for total
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = median_total),
             color = "black", linetype = "solid", size = 0.6, show.legend = FALSE) +
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = p25_total),
             color = "black", linetype = "dotted", size = 0.8, , show.legend = FALSE) +
  geom_vline(data = summary_stats %>% filter(emulator == "Total"),
             aes(xintercept = p75_total),
             color = "black", linetype = "dotted", size = 0.8, , show.legend = FALSE) +
  
  # Colors and labels
  scale_fill_manual(values = c("#0072B2","#D55E00", "grey"),
                    labels = c("MEWK_solar_IN_2030" = "Solar PV", 
                               "MEWK_onshore_IN_2030" = "Onshore",
                               
                               "Total Capacity" = "Total Capacity")) +
  
  scale_x_continuous(limits = c(0, 750)) +
  
  labs(
    x = "Capacity (GW)", 
    y = "Density",
    fill = "Technology",
    color = ''
  ) +
  guides(colour = "none") +
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



##################################################################

######## Emissions & demand

#################################################################

# stats for plot
summary_stats_3 <- df %>% subset(emulator %in% c('MEWE_IN_2030', 'MEWE_IN_2050') & 
                                   IN_cp == 'CP - None' & 
                                   IN_subsidy == 'Subsidy - None' & 
                                   IN_phase != 'Phaseouts - High' &
                                   !is.na(lead_total)) %>%
  
  # Create US policy factor variable
  mutate(
    year = factor(
      case_when(
        emulator == "MEWE_IN_2030" ~ "2030",
        emulator == "MEWE_IN_2050" ~ "2050"
      ),
      levels = c("2030",  "2050"))) %>%
  group_by(IN_phase, lead_total, emulator) %>%
  summarise(median_pred = median(prediction, na.rm = TRUE),
            .groups = 'drop') 

df %>% 
  subset(
    emulator %in% c('MEWE_IN_2030', 'MEWE_IN_2050') &
      IN_cp == 'CP - None' & 
      IN_subsidy == 'Subsidy - None' & 
      IN_phase != 'Phaseouts - High' &
      !is.na(lead_total)
  ) %>%
  
  mutate(
    year = factor(
      case_when(
        emulator == "MEWE_IN_2030" ~ "2030",
        emulator == "MEWE_IN_2050" ~ "2050"
      ),
      levels = c("2030", "2050")),
    demand = factor(
      case_when(
        demand == "Low demand" ~ "Low",
        demand == "High demand" ~ "High"
        ),
        levels = c("Low", "High")),
    year_demand = factor(
      paste(year, "-", demand),  # makes labels like "2030 - Low"
      levels = c("2030 - Low", "2030 - High", "2050 - Low", "2050 - High")
    )
  ) %>%
  
  ggplot(aes(x = prediction)) +
  
  # Median lines
  # geom_vline(aes(xintercept = 1000), color = "red", linetype = 'dashed') +
  # Density plots with fill/color by year and demand combo
  # geom_density(
  #   aes(
  #     fill = interaction(year, demand),
  #     color = 'black'),
  #   alpha = 0.9,
  #   adjust = 2, 
  #   size = 0.5, 
  #   linetype = 'solid'
  # ) +
  geom_density(
    aes(
      fill  = year_demand,
      color = 'black'
    ),
    alpha = 0.9,
    adjust = 2, 
    size = 0.5, 
    linetype = 'solid'
  ) +
  geom_vline(
    data = summary_stats_3, 
    aes(xintercept = median_pred, color = emulator),
    linetype = "dashed", size = 0.6, show.legend = FALSE
  ) +
  
  facet_grid(IN_phase ~ lead_total, labeller = label_value, scales = 'fixed') +
  
  labs(
    x = expression("Power Sector Emissions Levels (MtCO"[2]*"/year)"), 
    y = "Density",
    fill = "Year & Electricity Demand",
  ) +
  
  scale_fill_viridis_d(option = "C") +
  scale_color_viridis_d(option = "C") +
  scale_x_continuous(limits = c(0, NA)) + 
  
  scale_fill_manual(
    values = c(
      "2030 - Low"  = "#236192",
      "2030 - High" = "#3182bd",
      "2050 - Low"  = "#e34a33",
      "2050 - High" = "#fc8d59"
    )
  ) +
  scale_color_manual(
    values = c(
      "MEWE_IN_2030"  = "blue",
      "MEWE_IN_2050" = "red"
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











######################################################


########### Robustness over multiple targets


######################################################

# ### Capacity - solar & onshore 2030
# threshold <- 393
# summ_1 <- df %>% 
#   filter(emulator %in% c("MEWK_solar_IN_2030", "MEWK_onshore_IN_2030")) %>%
#   group_by(year, id, sample_id) %>% 
#   mutate(total_value = sum(prediction, na.rm = TRUE)) %>%
#   ungroup() %>%
#   group_by(`Policy Combination`, discount_rate, lead_total, demand) %>%
#   summarise(
#     total_obs = n()/2, # divide by 2 to correct for double count
#     breach_threshold = sum(total_value > threshold, na.rm = TRUE)/2,
#     target = "Capacity",
#     `Proportion achieving target` = round((breach_threshold / total_obs), 2),
#     mean = mean(total_value)) %>%
#   drop_na()

#### elec price
threshold <- 68
summ_2 <- df %>% 
  filter(emulator %in% c("MEWP_elec_price_IN_2030")) %>%
  group_by(`Policy Combination`, discount_rate,lead_total, demand) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction < threshold, na.rm = TRUE),
    target = "Electricity Price",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()


#####  emissions 
threshold <- 1000
summ_3 <- df %>% 
  filter(emulator %in% c("MEWE_IN_2030")) %>%
  group_by(`Policy Combination`, discount_rate, lead_total, demand) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction < threshold, na.rm = TRUE),
    target = "Emissions",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()

########## shares
#####  emissions 
threshold <- 0.55
summ_4 <- df %>% 
  filter(emulator %in% c("MEWS_renew_IN_2030")) %>%
  group_by(`Policy Combination`, discount_rate, lead_total, demand) %>%
  summarise(
    total_obs = n(),
    breach_threshold = sum(prediction > threshold, na.rm = TRUE),
    target = "Shares",
    `Proportion achieving target` = round((breach_threshold / total_obs), 2),
    mean = mean(prediction)) %>%
  drop_na()

# Can we do the plotting with just this df
summ <- rbind(summ_2, summ_3, summ_4)
# create column for ordering
summ <- summ %>% #filter(`Policy Combination` != 'No policy') %>%
  group_by(`Policy Combination`, discount_rate, lead_total, demand) %>%
  mutate(combined_prop = round(sum(`Proportion achieving target`), 2))



## Add together
plot_df <- summ %>% #filter(`Policy Combination` != 'No policy') %>%
  # e.g. emissions_prop
  bind_rows(summ) %>%
  mutate(
    # nicer labels
    `Policy Combination` = factor(`Policy Combination`,
                                  levels = c("No policy", "Subsidy & CP", "CP & Phaseout", 
                                             "Subsidy, CP & Phaseout", "Subsidy & Phaseout"),
                                  labels = c("Baseline",  "Sub-CP",  "CP-Phase", "Sub-CP-Phase", "Sub-Phase")),
    target = factor(target,
                    levels = c("Electricity Price", "Emissions", "Shares"), #"Capacity",
                    labels = c("Electricity Price","Emissions", "Shares")) #"Capacity",
  ) %>%
  ungroup() 

# Colour blind friendly pal
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



# Final plot for proportions over targets
ggplot(plot_df,
       aes(
         x    = `Policy Combination`,
         y    = `Proportion achieving target`,
         fill = target
       )) +
  geom_col(position = position_dodge(0.7), width = 0.7) +
  facet_grid(
    lead_total ~ discount_rate + demand,
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







#######################################################################




#### End of main plots



######################################################################





#############################################

## GLOBAL Emissions over time with phaseouts 

############################################
#FOT
# stats for plot
summary_stats_3 <- df %>% subset(emulator %in% c('MEWE_GBL_2030', 'MEWE_GBL_2050') & 
                                   !is.na(lead_total)) %>%
                                 
  # Create US policy factor variable
  mutate(
    year = factor(
      case_when(
        emulator == "MEWE_GBL_2030" ~ "2030",
        emulator == "MEWE_GBL_2050" ~ "2050"
      ),
      levels = c("2030",  "2050"))) %>%
  group_by(CN_phase, lead_total, emulator) %>%
  summarise(median_pred = median(prediction, na.rm = TRUE),
            p25_pred = quantile(prediction, 0.05, na.rm = TRUE),
            p75_pred = quantile(prediction, 0.95, na.rm = TRUE),
            .groups = 'drop') 



df %>% subset(emulator %in% c('MEWE_GBL_2030', 'MEWE_GBL_2050') & 
                                !is.na(lead_total)) %>%

  # Create US policy factor variable
  mutate(
    year = factor(
      case_when(
        emulator == "MEWE_GBL_2030" ~ "2030",
        emulator == "MEWE_GBL_2050" ~ "2050"
      ),
      levels = c("2030", "2050"))) %>%
  
  # Plot with ggplot
  ggplot(
    aes(x = prediction, fill = year
    )) +
  
  # Add median and percentile lines for total
 
  # geom_vline(data = summary_stats_3,
  #            aes(xintercept = p25_pred),
  #            color = "black", linetype = "dotted", size = 0.8, , show.legend = FALSE) +
  # geom_vline(data = summary_stats_3,
  #            aes(xintercept = p75_pred),
  #            color = "black", linetype = "dotted", size = 0.8, , show.legend = FALSE) +
  
  # Density plot for distribution
  geom_density(alpha = 0.5, adjust = 2) + 
  #geom_boxplot() +
  geom_vline(data = summary_stats_3, aes(xintercept = median_pred, color = emulator),
              linetype = "dashed", size = 0.6, show.legend = FALSE) +
  
  # geom_vline(
  #   data = summary_stats_3, 
  #   aes(xintercept = median_pred, color = emulator),
  #   linetype = "dashed", size = 0.6, show.legend = FALSE
  # ) +
  
  facet_grid(CN_phase  ~ lead_total, , labeller = label_value, scales = 'fixed') +
  # Define strong colors for each emulator
  labs(
    x = expression("Power Sector Emissions Levels (MtCO"[2]*"/year)"), 
    y = "Density",
    fill = "Year"
  ) + 
  scale_fill_viridis_d(option = "C") +
  scale_x_continuous(limits = c(0, 12500)) +
  scale_color_manual(
    values = c(
      "MEWE_GBL_2030"  = "purple",
      "MEWE_GBL_2050" = "black"
    )
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






##################################################




















# Inital bar plot cap & elec_price
ggplot(summ,
       aes(x = `Policy Combination`,
           y = `Proportion achieving target`,
           fill = target)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.7) +
  facet_grid(
    discount_rate ~ coal_p + demand,
    labeller = label_both,
    scales = "fixed"
  ) +
  labs(
    x = "Policy Combination",
    y = "Proportion achieving target",
    fill = "Target type",
    title = "Proportion of simulations over each target,\nby policy combo"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x   = element_text(angle = 45, hjust = 1),
    strip.text   = element_text(face = "bold"),
    legend.position = "bottom"
  )




# stats on peaks relative size - APPENDIX??
summary_df <-  df %>% 
    filter(
      emulator == 'MEWE_IN_2030',
      IN_phase == 'Phase - Mid',
      IN_cp == 'CP - None',
      `Solar Learning` == 'Low'
    ) %>%
    group_by(demand) %>%
    summarise(
      n = n(),
      mean = mean(prediction, na.rm = TRUE),
      sd = sd(prediction, na.rm = TRUE),
      min = min(prediction, na.rm = TRUE),
      p2.5 = quantile(prediction, 0.025, na.rm = TRUE),
      p25 = quantile(prediction, 0.25, na.rm = TRUE),
      median = median(prediction, na.rm = TRUE),
      p75 = quantile(prediction, 0.75, na.rm = TRUE),
      p97.5 = quantile(prediction, 0.975, na.rm = TRUE),
      max = max(prediction, na.rm = TRUE)
    ) %>%
    arrange(demand)

# Box plot - one policy by demand - APPENDIX??
df %>%
  filter(
    emulator == 'MEWE_IN_2030',
    IN_phase == 'Phase - Mid',
    IN_cp == 'CP - None',
    `Solar Learning` == 'Low'
  ) %>%
  ggplot(aes(x = demand, y = prediction, fill = demand)) +
  geom_boxplot(alpha = 0.6) +
  labs(
    x = "Demand Category",
    y = "Prediction",
    title = "Distribution of Predictions by Demand"
  ) +
  theme_minimal() +
  theme(legend.position = "none")









############################################################################


######## Leftovers


###########################################################################




## 4th plot Variation - one national policy package
df  %>% 
  group_by(year, id, sample_id) %>%  # Group by relevant variables
  mutate(total_value = sum(prediction, na.rm = TRUE)) %>% 
  ggplot(aes(x = total_value, color = demand, fill = demand)) +
  
  # Density plot with color for each Wind Learning group
  geom_density(alpha = 0.3) + 
  
  # Vertical reference line (optional)
  geom_vline(xintercept = 440, color = "red", linetype = "dotted", size = 2) +

  # Bar chart for Proportion over Target
  geom_bar(data = summ, aes(y = `Proportion over target`), 
           stat = "identity", width = 20, fill = "black", alpha = 0.5, inherit.aes = FALSE) +
  
  
  facet_grid(coal_p ~  `India Ambition`) +
  # Labels
  labs(
    # title = "2030 India Renewable Capacity Distributions",
    # subtitle = "by Policy Ambition - Coal Price - Electricity Demand",
    x = "Solar - Onshore Wind Capacity (GW)", 
    y = "Density",
    color = "Electricity Demand",
  ) +
  
  # Improved theme for clarity
  #theme_minimal() +
  theme(legend.position = "left")  




# Summarize the proportion data correctly
summ <- df %>%
  group_by(coal_p, `India Ambition`, demand) %>%
  summarise(`Proportion over target` = mean(`Proportion over target`), .groups = "drop")

ggplot(df_merged, aes(x = total_value, color = demand, fill = demand)) +
  
  # Density plot
  geom_density(alpha = 0.3) + 
  
  # Vertical reference line at 440 GW
  geom_vline(xintercept = 440, color = "red", linetype = "dotted", size = 1) +
  
  # Stacked bar plot for proportion over target
  geom_col(
    data = summ,
    aes(x = max(df_merged$total_value) + 50, 
        y = `Proportion over target` / max(`Proportion over target`) * max(density(df_merged$total_value)$y), 
        fill = demand),
    position = "stack",
    width = 40,
    alpha = 0.6,
    inherit.aes = FALSE
  ) +
  
  # Faceting
  facet_grid(coal_p ~ `India Ambition`) +
  
  # Labels
  labs(
    x = "Solar - Onshore Wind Capacity (GW)", 
    y = "Density",
    color = "Electricity Demand",
    fill = "Electricity Demand"
  ) +
  
  # Adjust y-scale for secondary axis (proportion over target)
  scale_y_continuous(
    name = "Density", 
    sec.axis = sec_axis(~ summ * max(`Proportion over target`), name = "Proportion Over Target", 
                        breaks = seq(0, 1, by = 0.2), labels = scales::percent)
  ) +
  
  # Ensure bars are not clipped
  coord_cartesian(clip = "off")




#######################

### Appendix

#######################




  
df %>% subset(EA_phase_pol == 0 & EA_price_pol == 0 & EA_cp_pol == 0 &
                IN_phase_pol == 0 & demand == "Medium Demand" &
                `Solar Learning` == "Medium" & 
                `Wind Learning` == "Medium" &
                coal_p == "Medium Price" &
                policy_combo %in% pol_grid) %>%
  mutate(facet_fill = factor(policy_combo)) %>%  # Unique fill variable for backgrounds
  
  ggplot(aes(x = total_value, fill = factor(policy_combo))) +
  
  # Background shading for each policy_combo
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, fill = facet_fill),
            alpha = 0.02, color = NA, show.legend = FALSE) +  # Adjusted alpha for better visibility
  # Density plot
  geom_density(color = "blue", fill = 'lightblue', alpha = 0.5) +
  # Density plot
  geom_vline(xintercept = 440, color = "red", linetype = "dotted", size = 1) +
  # Facet by CN and US policies
  facet_grid(CN_policy + IN_subsidy ~ US_policy + IN_cp, 
             labeller = label_value, switch = "y") +
  
  # Use different shades for each policy_combo
  scale_fill_manual(values = c("#FFDDC1", "#F4C2C2", "#C1E1C1", "#D4C1E1", 
                               "#C1D4E1", "#F1E1C1", "#E1C1D4", "#C2F4C2", 
                               "#FFDDC1", "#F4C2C2", "#C1E1C1", "#D4C1E1", 
                               "#C1D4E1", "#F1E1C1", "#E1C1D4", "#C2F4C")) + 
  
  labs(
    x = "Solar-Onshore Wind Capacity (GW)", 
    y = "Density"
  ) 

#########################

## Labellers

#######################



# Can we have labels in other positions?
custom_labeller <- function(variable, value) {
  return(paste(variable, ":", value))
}

facet_grid(rows = vars(CN_cp), cols = vars(US_cp), labeller = custom_labeller)

custom_labeller <- function(variable, value) {
  if (variable == "US_cp" & value == 1)
    {
    return(paste("US:", "Carbon Tax"))
  } else if (variable == "CN_cp") {
    return(paste("CN:", value))
  } else {
    return(value)  # Default case
  }
}

#################

## Individual policy analysis


## 1st plot Overall distribution
df <- plot_data %>% subset(#elec_demand >= 0.3 & elec_demand < 0.6 &
  `India Ambition` == "Mid Ambition"
    #`India Ambition` == "High Ambition"
) 

# Compute quantiles for each facet group
quantiles_df <- df %>%
  group_by(`India Ambition`) %>%
  summarise(
    q_low = quantile(total_value, 0.025),
    q_high = quantile(total_value, 0.975),
    .groups = "drop"
  )

# Plot density with facet-specific quantiles
df  %>% subset(`India Ambition` == "Mid Ambition") %>%
  ggplot(aes(x = total_value)) +
  geom_density(fill = "blue", alpha = 0.4) +
  geom_vline(data = quantiles_df, aes(xintercept = q_low), linetype = "dashed", color = "white", size = 1) +
  geom_vline(data = quantiles_df, aes(xintercept = q_high), linetype = "dashed", color = "white", size = 1) +
  labs(title = "Desnity plot of Mid Ambition",
       x = "Energy Capacity",
       y = "Density")# +
  facet_grid(coal_p~demand)
  
# REorder levels for geom_col  
df_merged$demand <- factor(df_merged$demand, levels = c("Low Demand", "Medium Demand", "High Demand"))
summ$demand <- factor(summ$demand, levels = levels(df_merged$demand))  # Reverse order for stacked bars


## 2nd Plot Comparison, doesn't need to have df merged now?
ggplot(df_merged, aes(x = total_value, color = demand, fill = demand)) +
  # Density plot
  geom_density(alpha = 0.3) + 
  
  # Vertical reference line
  geom_vline(xintercept = 440, color = "red", linetype = "dotted", size = 1) +
  
  # Faceting
  facet_grid(coal_p ~ `India Ambition`) +
  
  # Axes adjustments
  scale_y_continuous(name = "Density") +
  
  # Labels and theme
  labs(
    x = "Solar - Onshore Wind Capacity (GW)", 
    y = "Density",
    color = "Electricity Demand",
    fill = "Electricity Demand"
  ) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 13),  # Increase size of facet labels
    axis.title.x = element_text(size = 16),  # Increase size of x-axis title
    axis.title.y = element_text(size = 16)  # Increase size of y-axis title
  ) 

## 3rd Plot Bar chart#
summ %>% subset( `India Ambition` != "No Ambition") %>%
ggplot(aes(x = demand, y = `Proportion over target`, fill = demand)) +
  geom_col(position = "dodge", width = 0.6, alpha = 0.7) +  # Side-by-side bars
  facet_grid(coal_p ~ `India Ambition`) +  # Keep the same faceting structure
  
  # Formatting the y-axis
  scale_y_continuous(
    name = "Proportion Over Target",
    labels = scales::percent_format(),  # Format as percentage
    limits = c(0, 1)  # Ensure the range goes from 0 to 1
  ) +
  
  # Labels and theme
  labs(
    x = "Electricity Demand Category", 
    y = "Proportion Over Target",
    fill = "Electricity Demand"
  ) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 13),  # Increase facet label size
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(angle = 5) 
    
    )

#################################


max_density <- 0.005
#max_density <- max(density(df_merged$total_value)$y)


# Get max proportion value (should be 1.0 at most)
max_proportion <- max(summ$`Proportion over target`, na.rm = TRUE)

ggplot(df_merged, aes(x = total_value, color = demand, fill = demand)) +
  # Density plot
  geom_density(alpha = 0.3) + 
  
  # Vertical reference line
  geom_vline(xintercept = 440, color = "red", linetype = "dotted", size = 1) +
  
  # Stacked bar plot for proportion over target
  geom_col(
    data = summ,
    aes(x = max(df_merged$total_value) + 50, 
        y = (`Proportion over target` / max_proportion) * max_density, 
        fill = demand),
    width = 40,
    alpha = 0.6,
    inherit.aes = FALSE
  ) +
  
  # Faceting
  facet_grid(coal_p ~ `India Ambition`) +
  
  # Axes adjustments
  scale_y_continuous(
    name = "Density",
    sec.axis = sec_axis(
      ~ . / max_density * max_proportion,  # Correct transformation
      name = "Proportion Over Target",
      breaks = c(0, max_proportion*1.5, max_proportion *3),  # Ensures even spacing
      labels = c("0%", "50%", "100%")
    )
  ) +
  
  # Labels and theme
  labs(
    x = "Solar - Onshore Wind Capacity (GW)", 
    y = "Density",
    color = "Electricity Demand",
    fill = "Electricity Demand"
  ) +
  # Customizing text sizes
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 13),  # Increase size of facet labels
    axis.title.x = element_text(size = 16),  # Increase size of x-axis title
    axis.title.y = element_text(size = 16)  # Increase size of y-axis title
    # axis.text.x = element_text(size = 12),   # Increase size of x-axis tick labels
    # axis.text.y = element_text(size = 12)    # Increase size of y-axis tick labels
  ) 
  

  

################################

#### BASIS EMULATOR TIME SERIES

################################

# par(mfrow = c(2, 3), mar = c(2,2,2,2)) # big plot for higher no. vars
# 
# 
# plot_data %>% subset(year %in% c(2030, 2050)) %>%
# ggplot(aes(x = year, y = prediction, group = year)) +
#   geom_boxplot() +
#   geom_line() +
#   labs(title = "Distribution of Values in 2050",
#        x = "Scenario",
#        y = "Emissions") +
#   theme_minimal() +
#   facet_grid(emulator~US_pol + ROW_pol, scales = 'free')
# 
# 
# ####### PLAYYY
# # plot_data %>% subset(emulator == 'MEWE_GBL' & Year > 2022) %>% 
# # ggplot(aes(x = Year, y = Value)) +
# #   stat_density_2d(aes(fill = after_stat(density)), geom = "raster", contour = FALSE) +
# #   scale_fill_viridis_c() +
# #   labs(title = "Density of Trajectories Over Time") +
# #   theme_minimal() +
# #   facet_grid(US_pol ~ ROW_pol)
# 
# # Spaghetti plot
# plot_data %>% subset(emulator == 'MEWE_GBL' & Year > 2022 & 
#                        ROW_pol == 'ROW mid-ambition') %>% 
#   ggplot(aes(x = Year, y = Value, group = sample_id)) +
#   geom_line(alpha = 0.05, color = "blue") +
#   labs(title = "Emissions over Time - Global",
#        y = 'Emissions') +
#   theme_minimal() +
#   facet_grid(US_pol ~ ROW_pol)
# 
# 
# plot_data %>% subset(emulator == 'MEWW_onshore_GBL' & Year > 2022 &
#                      ROW_pol == 'ROW baseline') %>% 
#   ggplot(aes(x = Year, y = Value, group = sample_id)) +
#   geom_line(alpha = 0.05, color = "blue") +
#   labs(title = "Onshore Capacity - Global",
#        y = 'Capacity (GW)') +
#   theme_minimal() +
#   facet_grid(US_pol ~ ROW_pol)
# 
# plot_data %>% subset(emulator == 'MEWW_solar_GBL' & Year > 2022 &
#   ROW_pol == 'ROW baseline') %>%  
#   ggplot(aes(x = Year, y = Value, group = sample_id)) +
#   geom_line(alpha = 0.05, color = "blue") +
#   labs(title = "Solar Capacity - Global",
#        y = 'Capacity (GW)') +
#   theme_minimal() +
#   facet_grid(US_pol ~ ROW_pol)
# 
# plot_data %>% subset(emulator != 'MEWE_GBL' & Year > 2022) %>%  
#   ggplot(aes(x = Year, y = Value, group = sample_id)) +
#   geom_line(alpha = 0.05, color = "blue") +
#   labs(title = "Solar Capacity - Global",
#        y = 'Capacity (GW)') +
#   theme_minimal() +
#   facet_grid(US_pol ~ emulator + ROW_pol)
# 
# 
# plot_data %>% subset(Year %in% c(2030, 2040,2050) & 
#                                    emulator == 'MEWE_GBL' &
#                        ROW_pol == 'ROW mid-ambition')  %>%
#   ggplot(aes(x = Year, y = Value, group = Year)) +
#   # geom_jitter(aes(color = "blue", alpha = 0.001)) +
#   geom_boxplot() +
#   labs(title = "Distribution of Global Capacities",
#                             x = "Year",
#                             y = "Emissions") +
#                        facet_grid(emulator ~ US_pol + ROW_pol, scales = 'free')
# 
# stats_df <- plot_data %>%
#   filter(emulator == 'MEWW_solar_GBL',Year %in% c(2030, 2040, 2050),
#            ROW_pol == 'ROW baseline',
#          US_pol == 'US baseline') %>%
#   group_by(emulator, US_pol, Year, discount_rate, lr_solar) %>%
#   summarise(
#     mean = mean(Value, na.rm = TRUE),
#     q25 = quantile(Value, 0.25, na.rm = TRUE),
#     q75 = quantile(Value, 0.75, na.rm = TRUE),
#     .groups = 'drop'
#   )
# 
# 
# plot_data %>% subset(emulator == 'MEWW_onshore_GBL' & 
#                        Year %in% c(2030, 2040,2050) &
#                        ROW_pol == 'ROW baseline') %>%
#   ggplot(aes(x = Value, color = Value)) +
#   geom_density(alpha = 0.5, fill = "steelblue") +
#   geom_vline(data = stats_df, aes(xintercept = mean), linetype = "dashed", color = "black") +
# geom_vline(data = stats_df, aes(xintercept = q25), linetype = "dotted", color = "blue") +
# geom_vline(data = stats_df, aes(xintercept = q75), linetype = "dotted", color = "blue") +
#   geom_rug(sides = "b", alpha = 0.2) +
#   labs(title = 'Onshore Capacity Global', 
#        x = 'Capacity (GW)') +
#   #geom_histogram() +
#   facet_grid(US_pol + ROW_pol ~ Year)

#####################################################

##### DISCR & LR

####################################################

# stats_df <- plot_data %>%
#   filter(emulator == 'MEWW_solar_GBL',year %in% c(2030, 2050),
#          ROW_pol == 'ROW baseline') %>%
#   group_by(emulator, US_pol, Year, discount_rate, lr_solar) %>%
#   summarise(
#     mean = mean(Value, na.rm = TRUE),
#     q25 = quantile(Value, 0.25, na.rm = TRUE),
#     q75 = quantile(Value, 0.75, na.rm = TRUE),
#     .groups = 'drop'
#   )
# 
# plot_data %>% subset(emulator == 'MEWW_solar_GBL' & 
#                        Year %in% c(2030,2040,2050) &
#                        ROW_pol == 'ROW baseline') %>%
#   ggplot(aes(x = Value, group = discount_rate)) +
#   geom_density(aes(fill = discount_rate),alpha = 0.5) +
#   geom_vline(data = stats_df, aes(xintercept = mean, color = discount_rate), 
#              linetype = "dashed", size = 0.8) +   # geom_vline(data = stats_df, aes(xintercept = q25), linetype = "dotted", color = "blue") +
#   labs(title = 'Solar Capacity Global', 
#        x = 'Capacity (GW)') +
#   facet_grid(US_pol + lr_solar ~ Year)
# 
# 
# plot_data %>% subset(emulator == 'MEWW_onshore_GBL' & 
#                        Year %in% c(2030, 2040,2050) &
#                        ROW_pol == 'ROW baseline') %>%
#   ggplot(aes(x = Value)) +
#   geom_density(aes(fill = discount_rate),alpha = 0.5) +
#   #geom_vline(data = stats_df, aes(xintercept = mean, color = discount_rate), 
#              #linetype = "dashed", size = 0.8) +  
#   # geom_vline(data = stats_df, aes(xintercept = q25), linetype = "dotted", color = "blue") +
#   # geom_vline(data = stats_df, aes(xintercept = q75), linetype = "dotted", color = "blue") +
#   labs(title = 'Onshore Capacity Global', 
#        x = 'Capacity (GW)') +
#   #geom_histogram() +
#   facet_grid(US_pol ~ Year + lr_wind)
# 
# 
# plot_data %>%
#   filter(emulator != 'MEWE_GBL', Year %in% c(2050), ROW_pol == "ROW baseline") %>%
#   ggplot(aes(x = Value, fill = US_pol)) +
#   geom_density(alpha = 0.6) +
#   geom_vline(data = stats_df[stats_df$Year == 2050,], aes(xintercept = mean, group = US_pol),
#              linetype = "dashed", color = "blue", size = 0.8) +
#   geom_vline(data = stats_df[stats_df$Year == 2050,], aes(xintercept = q25, group = US_pol),
#              linetype = "dotted", color = "black", size = 0.6) +
#   geom_vline(data = stats_df[stats_df$Year == 2050,], aes(xintercept = q75, group = US_pol),
#              linetype = "dotted", color = "black", size = 0.6) +
#   facet_grid(US_pol ~ emulator + Year, scales = "free")
# 
# plot_data %>%
#   filter(emulator != 'MEWE_GBL', Year %in% c(2030, 2040, 2050), ROW_pol == "ROW baseline") %>%
#   ggplot(aes(x = Value, fill = US_pol)) +
#   geom_density(alpha = 0.6) +
#   geom_vline(data = stats_df, aes(xintercept = mean, group = US_pol),
#              linetype = "dashed", color = "blue", size = 0.8) +
#   geom_vline(data = stats_df, aes(xintercept = q25, group = US_pol),
#              linetype = "dotted", color = "black", size = 0.6) +
#   geom_vline(data = stats_df, aes(xintercept = q75, group = US_pol),
#              linetype = "dotted", color = "black", size = 0.6) +
#   facet_grid(US_pol ~ emulator + Year, scales = "fixed")
# 
# 
# #### Checking for combinations policies
# # Define variables to check
# selected_vars <- c("EA_cp_pol", "EA_phase_pol", "EA_price_pol",
#                    "CN_cp_pol","CN_phase_pol","CN_price_pol", 
#                    "US_cp_pol","US_phase_pol","US_price_pol")
# # Get unique combinations
# unique_combinations <- plot_data_instrument %>% distinct(across(all_of(selected_vars)))
# 
# 
# ### Filter for instrument analysis
# 
# 
# %>%
group_by(discount_rate, coal_p, demand) %>%
  mutate(
    `Policy Combination` = factor(
      `Policy Combination`,
      levels = unique(`Policy Combination`[order(combined_prop)])
    )
  ) %>%
  ungroup()


# 1) Extract just the Average rows, order them by combined_prop,
# #    and give them an increasing facet_id:
# facet_order <- plot_df %>% 
#   filter(target == "Average") %>% 
#   group_by(demand, discount_rate, coal_p) %>%
#   arrange(combined_prop, .by_group = FALSE) %>%
#   mutate(facet_id = row_number()) %>%
#   ungroup() %>%
#   dplyr::select(discount_rate, coal_p, demand, `Policy Combination`, facet_id)
# 
# # 2) Join that small table back onto the full plot_df2:
# plot_df <- plot_df %>%
#   left_join(facet_order,
#             by = c("discount_rate", "coal_p", "demand", "Policy Combination")) %>%
#   ungroup()   %>%
#   group_by(discount_rate, coal_p, demand) %>%
#   mutate(
#     `Policy Combination` = factor(
#       `Policy Combination`,
#       levels = unique(`Policy Combination`[order(facet_id)])
#     )
#   ) %>%
#   ungroup()
# 
# 
# 
# # 1) Re‐level Policy Combination by facet_id within each facet‐group
# plot_df_ordered <- plot_df %>%
#   group_by(discount_rate, coal_p, demand) %>%
#   # sort each block by facet_id, then capture that order as the new factor levels
#   arrange(facet_id, .by_group = TRUE) %>%
#   mutate(
#     `Policy Combination` = factor(
#       `Policy Combination`,
#       levels = unique(`Policy Combination`)
#     )
#   ) %>%
#   ungroup()

#   

 

