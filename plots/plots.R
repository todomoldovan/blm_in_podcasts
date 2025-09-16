library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(scales)
library(readr)
library(lubridate)
library(tidyr)
library(grid)  

emo_counts <- read_csv("../data/emotion_counts.csv")

custom_order <- c("anger", "disgust", "fear", "sadness", "joy", "optimism", "love", "caring")

emo_counts_filtered <- emo_counts %>%
  filter(emotion %in% custom_order) %>%
  mutate(emotion = factor(emotion, levels = custom_order)) %>%
  arrange(emotion) %>%
  mutate(
    total = count_0 + count_1,
    count_0_pct = count_0 / total * 100,
    count_1_pct = count_1 / total * 100
  )

###################################################################################################################

df <- tibble::tibble(
  Category = c("Problem-solution", "Call-to-action", "Intention", "Execution"),
  Count    = c(1605020, 261979, 219763, 93212) 
) %>%
  mutate(
    Category = factor(Category, levels = c("Problem-solution", "Call-to-action", "Intention", "Execution")),
    Percent  = Count / sum(Count) * 100   
  )

p <- ggplot(df, aes(x = Percent, y = Category, fill = Category)) +
  geom_col(width = 0.7) +
  scale_x_continuous(
    labels = function(x) sprintf("%d", x),  
    breaks = seq(0, 100, by = 10),          
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_discrete(limits = rev(levels(df$Category))) +
  scale_fill_brewer(palette = "Spectral") +
  guides(fill = "none") +
  labs(
    x = "Average % per episode",
    y = NULL
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(size = 30, color = "black"),
    axis.text.y = element_text(size = 30, color = "black", margin = margin(r = 6)),
    axis.title.x = element_text(size = 30, color = "black"),
    axis.title.y = element_text(size = 30, color = "black")
  )

ggsave("../plots/class_percentages.pdf", p, width = 10, height = 5)

###################################################################################################################

race <- read_csv("../data/race_epIDs.csv")
all_episodes <- read_csv("../data/all_episodes_dates.csv")

race <- race %>%
  mutate(episodeDateLocalized = as_datetime(episodeDateLocalized / 1000))

all_episodes <- all_episodes %>%
  mutate(episodeDateLocalized = as_datetime(episodeDateLocalized / 1000))

race_counts <- race %>%
  group_by(date = as_date(episodeDateLocalized)) %>%
  summarise(race_count = n(), .groups = "drop")

all_counts <- all_episodes %>%
  group_by(date = as_date(episodeDateLocalized)) %>%
  summarise(all_count = n(), .groups = "drop")

spectral_colors <- brewer.pal(11, "Spectral")[c(10, 2)]

weekend_dates <- as.Date(c(
  "2020-05-30",
  "2020-06-06",
  "2020-06-13",
  "2020-06-20",
  "2020-06-27"
))

weekend_points <- race_counts %>%
  filter(date %in% weekend_dates)

annotations <- data.frame(
  date = as.Date(c(
    "2020-05-25", "2020-06-02", "2020-06-19"
  )),
  label = c(
    "Murder of George Floyd", "Blackout\nTuesday", "Juneteenth"
  )
)

yrng  <- diff(range(race_counts$race_count, na.rm = TRUE))
y_max <- max(race_counts$race_count, all_counts$all_count, na.rm = TRUE)

set.seed(42) 

ann <- annotations %>%
  left_join(race_counts, by = "date") %>%
  rename(y = race_count) %>%
  #drop_na(y) %>%  
  mutate(
    mult  = runif(n(), min = 0.6, max = 0.8),
    y_lab = y + mult * yrng,
    x_lab = date + rep(c(-1, 1), length.out = n()) * 0.3
  )

p <- ggplot() +
  geom_line(
    data = all_counts,
    aes(date, all_count, color = "All podcasts"),
    linewidth = 0.8
  ) +
  geom_line(
    data = race_counts,
    aes(date, race_count, color = "Racial discussions"),
    linewidth = 1
  ) +
  geom_point(
    data = weekend_points,
    aes(date, race_count, shape = "Weekend of protests"),
    color = spectral_colors[1], size = 4
  ) +
  geom_point(
    data = ann,
    aes(date, y, shape = "Important events"),
    color = "black", size = 3
  ) +
  geom_segment(
    data = ann,
    aes(x = date, xend = date, y = y, yend = y_lab),
    linetype = "dashed", linewidth = 0.3,
    show.legend = FALSE
  ) +
  geom_text(
    data = ann,
    aes(x = x_lab, y = y_lab, label = label),
    vjust = -0.2, size = 8,
    show.legend = FALSE
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.02))) +
  coord_cartesian(ylim = c(0, y_max), clip = "off") +
  scale_x_date(labels = function(d) sub(" 0", " ", format(d, "%b %d"))) +
  
  scale_color_manual(
    values = c("All podcasts" = "grey70",
               "Racial discussions" = spectral_colors[1])
  ) +
  scale_shape_manual(
    values = c("Weekend of protests" = 15, # square
               "Important events" = 16) # circle
  ) +
  
  labs(
    x = "Date",
    y = "Number of episodes",
    color = NULL,
    shape = NULL
  ) +
  guides(
    color = guide_legend(order = 1),
    shape = guide_legend(order = 2)
  ) +
  theme_minimal(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.text = element_text(size = 30, color = "black"),
    axis.text.x = element_text(size = 30, color = "black"),
    axis.text.y = element_text(size = 30, color = "black"),
    axis.title.x = element_text(size = 30, color = "black", margin = margin(t = 6)),
    axis.title.y = element_text(size = 30, color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(4, "pt"),
    plot.margin = margin(10, 100, 10, 10)
  )

ggsave("../plots/time_series.pdf", p, width = 10, height = 8)

###################################################################################################################

library(ggplot2)
library(dplyr)
library(tidyr)
# library(hrbrthemes) 

# odds_ratio <- read.csv("../data/odds_ratio.csv")
odds_ratio_problem_solution <- read.csv("../data/odds_ratio_problem_solution.csv")
odds_ratio_call_to_action   <- read.csv("../data/odds_ratio_call_to_action.csv")
odds_ratio_intention        <- read.csv("../data/odds_ratio_intention.csv")
odds_ratio_execution        <- read.csv("../data/odds_ratio_execution.csv")

names(odds_ratio_problem_solution)[names(odds_ratio_problem_solution) == "category"] <- "emotion"
names(odds_ratio_call_to_action)[names(odds_ratio_call_to_action) == "category"]     <- "emotion"
names(odds_ratio_intention)[names(odds_ratio_intention) == "category"]               <- "emotion"
names(odds_ratio_execution)[names(odds_ratio_execution) == "category"]               <- "emotion"

data <- odds_ratio_problem_solution %>%
  select(emotion, odds_ratio) %>%
  rename(value1 = odds_ratio) %>%
  left_join(odds_ratio_call_to_action %>% select(emotion, odds_ratio) %>% rename(value2 = odds_ratio), by = "emotion") %>%
  left_join(odds_ratio_intention %>% select(emotion, odds_ratio) %>% rename(value3 = odds_ratio), by = "emotion") %>%
  left_join(odds_ratio_execution %>% select(emotion, odds_ratio) %>% rename(value4 = odds_ratio), by = "emotion")

data <- data %>%
  mutate(emotion = factor(emotion, levels = custom_order)) %>%
  arrange(emotion)

data_long <- data %>%
  pivot_longer(cols = starts_with("value"), names_to = "variable", values_to = "odds_ratio_value") %>%
  group_by(emotion) %>%
  mutate(min_value = min(odds_ratio_value), max_value = max(odds_ratio_value)) %>%
  ungroup()

p <- ggplot(data_long, aes(x = emotion, y = odds_ratio_value, color = variable)) +
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = 0.5, ymax = 1.5,
           fill = "grey70", alpha = 0.15) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "black") +
  geom_point(size = 5) +
  coord_flip() +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "none",
    axis.text.x  = element_text(size = 30, color = "black"),
    axis.text.y  = element_text(size = 30, color = "black", margin = margin(r = 6)),
    axis.title.x = element_text(size = 30, color = "black"),
    axis.title.y = element_text(size = 30, color = "black"),
    strip.text   = element_text(size = 30)
  ) +
  scale_color_brewer(palette = "Spectral") +
  facet_wrap(
    ~ variable, nrow = 1,
    labeller = as_labeller(c(
      value1 = "Problem\nsolution",
      value2 = "Call to\naction",
      value3 = "Intention",
      value4 = "Execution"
    ))
  ) +
  xlab(NULL) + ylab("Odds ratio") +
  scale_x_discrete(limits = rev(levels(data_long$emotion)))

ggsave("../plots/lolliplot_emotions.pdf", p, width = 10, height = 5)

###################################################################################################################

odds_ratio_problem_solution <- read.csv("../data/odds_ratio_problem_solution_simple.csv")
odds_ratio_call_to_action   <- read.csv("../data/odds_ratio_call_to_action_simple.csv")
odds_ratio_intention        <- read.csv("../data/odds_ratio_intention_simple.csv")
odds_ratio_execution        <- read.csv("../data/odds_ratio_execution_simple.csv")

names(odds_ratio_problem_solution)[names(odds_ratio_problem_solution) == "category"] <- "emotion"
names(odds_ratio_call_to_action)[names(odds_ratio_call_to_action) == "category"]     <- "emotion"
names(odds_ratio_intention)[names(odds_ratio_intention) == "category"]               <- "emotion"
names(odds_ratio_execution)[names(odds_ratio_execution) == "category"]               <- "emotion"

data <- odds_ratio_problem_solution %>%
  select(emotion, odds_ratio) %>%
  rename(value1 = odds_ratio) %>%
  left_join(odds_ratio_call_to_action %>% select(emotion, odds_ratio) %>% rename(value2 = odds_ratio), by = "emotion") %>%
  left_join(odds_ratio_intention %>% select(emotion, odds_ratio) %>% rename(value3 = odds_ratio), by = "emotion") %>%
  left_join(odds_ratio_execution %>% select(emotion, odds_ratio) %>% rename(value4 = odds_ratio), by = "emotion")

data_long <- data %>%
  pivot_longer(cols = starts_with("value"), names_to = "variable", values_to = "odds_ratio_value") %>%
  group_by(emotion) %>%
  mutate(min_value = min(odds_ratio_value), max_value = max(odds_ratio_value)) %>%
  ungroup()

p <- ggplot(data_long) +
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = 0.5, ymax = 1.5,
           fill = "grey70", alpha = 0.15) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "black") +
  geom_point(aes(x = emotion, y = odds_ratio_value, color = variable), size = 5) +
  coord_flip() +
  scale_color_brewer(
    palette = "Spectral",
    labels = c(
      "value1" = "Problem\nsolution",
      "value2" = "Call to\naction",
      "value3" = "Intention",
      "value4" = "Execution"
    )
  ) +
  guides(color = guide_legend(title = NULL, nrow = 1, byrow = TRUE)) +
  xlab(NULL) + ylab("Odds ratio") +
  scale_x_discrete(limits = rev(levels(data_long$emotion))) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "top",
    legend.text  = element_text(size = 30, color = "black"),
    axis.text.x  = element_text(size = 30, color = "black"),
    axis.text.y  = element_text(size = 30, color = "black", margin = margin(r = 6)),
    axis.title.x = element_text(size = 30, color = "black"),
    axis.title.y = element_text(size = 30, color = "black")
  )

ggsave("../plots/lolliplot_sentiment.pdf", p, width = 10, height = 5)




