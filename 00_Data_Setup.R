###--------------------------------------------------------------------------###
###   00_Data_Setup.R                                                        ###
###   Shared data foundation — sourced by both Part 1 and Part 2             ###
###                                                                          ###
###   Author: Keziah Vickraman                                               ###
###   Capstone: NBA Achilles Load Management Study                           ###
###                                                                          ###
###   This file does four things and nothing else:                           ###
###     1. Load packages                                                     ###
###     2. Import raw data (hoopR + injury CSV + ESPN scrape)                ###
###     3. Build shared clean objects used by both Part 1 and Part 2         ###
###     4. Define theme_nba() used in all visualisations                     ###
###                                                                          ###
###--------------------------------------------------------------------------###

rm(list = ls()
   )

###==========================================================================###
###  1. PACKAGES                                                               ###
###==========================================================================###

pacman::p_load(
  # Core
  tidyverse,
  lubridate,
  glue,
  zoo,
  # NBA data
  hoopR,
  rvest,
  tidymodels,
  ranger,
  xgboost,
  glmnet,
  doParallel,
  future,
  workflowsets,
  parsnip,
  recipes,
  workflows,
  tune,
  yardstick,
  rsample,
  broom,
  # Causal / regression tables
  jtools,
  huxtable,
  # Visualisation
  showtext,
  scales,
  gridExtra,
  ggthemes,
  DT,
  plotly
)

###==========================================================================###
###  2. CUSTOM NBA THEME                                                       ###
###==========================================================================###

# NBA brand palette:
#   #CE1141  red      #17408B  blue
#   #FFFFFF  white    #E0E0E0  light grey

font_add_google("Roboto Condensed", "roboto")
showtext_auto()

theme_nba <- function() {
  theme_minimal(base_family = "sans") +
    theme(
      plot.background  = element_rect(fill = "#FFFFFF", color = NA),
      panel.grid.major = element_line(color = "#E0E0E0"),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = 16, color = "black"),
      plot.subtitle    = element_text(size = 12, color = "gray30"),
      plot.caption     = element_text(size = 8,  color = "black", face = "italic",
                                      hjust = 0),
      axis.title       = element_text(face = "bold", size = 10),
      axis.text        = element_text(size = 9),
      legend.position  = "top",
      legend.direction = "horizontal",
      legend.title     = element_text(face = "bold", size = 10, color = "black"),
      legend.text      = element_text(size = 9, color = "black")
    )
}

###==========================================================================###
###  3. IMPORT                                                                 ###
###==========================================================================###

## 3a. NBA box scores via hoopR (2002–2025) --------------------------------
# Pull once — expensive. Comment out after first run and load the CSV instead.
# Load from RData if available (Shiny deployment)
# Raw CSVs are not included in deployment bundle
if (exists("all_seasons_box")) {
  message("Data already loaded — skipping CSV reads")
} else if (file.exists("00_Shared_Data.RData")) {
  load("00_Shared_Data.RData")
} else {
  message("Warning: raw data not available — some objects may be missing")
}
# seasons <- 2002:2025
# 
# regular_box <-
#   map_dfr(seasons,
#           ~ load_nba_player_box(seasons = .x, season_types = "Regular Season"))
# 
# playoff_box <-
#   map_dfr(seasons,
#           ~ load_nba_player_box(seasons = .x, season_types = "Playoffs"))
# 
# all_seasons_box <- bind_rows(regular_box, playoff_box)
# write_csv(all_seasons_box, "all_seasons_NBA_2002_2025.csv")

# After first pull, just load the CSV:
all_seasons_box <- read_csv("all_seasons_NBA_2002_2025.csv")

## 3b. Kaggle / ProSports injury log (1951–2023) ---------------------------
injuries_raw <- read_csv("NBA Player Injury Stats(1951 - 2023).csv")

## 3c. ESPN 2024–2025 injury log (web scrape) ------------------------------
# Wrap in tryCatch so the script doesn't break if ESPN changes their HTML

espn_injury_CLEAN <- tryCatch({

  bad_values <- c("EST. RETURN DATE", "TBD", "Out", "None", "-", "—", "N/A", "")

  "https://www.espn.com.sg/nba/injuries" %>%
    read_html() %>%
    html_node(paste0(
      "#fittPageContainer > div.pageContent > ",
      "div.page-container.cf > div > div > section > div > section"
    )) %>%
    html_table() %>%
    rename(return_date_est_raw = `EST. RETURN DATE`) %>%
    filter(!return_date_est_raw %in% bad_values) %>%
    mutate(
      return_date_est_md = parse_date_time(return_date_est_raw, orders = "d b"),
      return_date_full   = update(return_date_est_md, year = 2025)
    ) %>%
    filter(!is.na(return_date_est_md)) %>%
    select(-return_date_est_md)

}, error = function(e) {
  message("ESPN scrape failed — using empty placeholder. Error: ", e$message)
  tibble(NAME = character(), COMMENT = character(), return_date_full = as.Date(NA))
})

## 3d. Manually curated 2024–2025 Achilles rupture cases -------------------
# These 8 confirmed ruptures are used in Part 1 (EDA + DiD context) only.
# They do NOT feed into the Part 2 prediction target.

new_achilles_cases_2425 <- tribble(
  ~player_name,        ~injury_date,  ~injury_desc,                              ~injury_type, ~injury_severity,
  "Tyrese Haliburton", "2025-06-22",  "reported torn right Achilles tendon",     "rupture",    "high",
  "Jayson Tatum",      "2025-05-12",  "reported torn right Achilles tendon",     "rupture",    "high",
  "Dejounte Murray",   "2025-01-31",  "reported torn right Achilles tendon",     "rupture",    "high",
  "Damian Lillard",    "2025-04-27",  "reported torn left Achilles tendon",      "rupture",    "high",
  "Isaiah Jackson",    "2024-11-01",  "reported torn right Achilles tendon",     "rupture",    "high",
  "James Wiseman",     "2024-10-23",  "reported torn left Achilles tendon",      "rupture",    "high",
  "Dru Smith",         "2024-12-23",  "reported ruptured left Achilles tendon",  "rupture",    "high",
  "DaRon Holmes II",   "2024-07-12",  "reported torn right Achilles tendon",     "rupture",    "high"
) %>%
  mutate(
    injury_date             = ymd(injury_date),
    achilles_rupture_binary = 1L
  )

###==========================================================================###
###  4. SHARED CLEAN OBJECTS                                                   ###
###==========================================================================###

## 4a. Confirmed Achilles rupture dataset (1990–2025) ----------------------
# Used in Part 1 for EDA visualisations and DiD.
# Filtered to confirmed ruptures only (keywords: tear, repair, surgery etc.)

achilles_ruptures_HISTORICAL <-
  injuries_raw %>%
  rename(
    injury_date   = Date,
    player_name_1 = Relinquished,
    player_name_2 = Acquired,
    injury_desc   = Notes
  ) %>%
  mutate(
    player_name = coalesce(player_name_1, player_name_2),
    player_name = str_squish(str_remove_all(player_name, "\\(.*\\)")),
    injury_date = as.Date(injury_date)
  ) %>%
  filter(
    !is.na(player_name),
    str_detect(tolower(injury_desc), "achilles"),
    year(injury_date) >= 1990
  ) %>%
  mutate(
    injury_type = case_when(
      str_detect(injury_desc,
                 regex("rupture(d)?|tear|torn|repair|surgery|reconstruction",
                       ignore_case = TRUE)) ~ "rupture",
      str_detect(injury_desc,
                 regex("tendinitis|tendonitis",
                       ignore_case = TRUE)) ~ "tendinitis",
      str_detect(injury_desc,
                 regex("sore|irritation|tightness",
                       ignore_case = TRUE)) ~ "sore",
      TRUE ~ "unclear"
    ),
    injury_severity         = case_when(
      injury_type == "rupture"    ~ "high",
      injury_type == "tendinitis" ~ "medium",
      injury_type == "sore"       ~ "low",
      TRUE                        ~ "none"
    ),
    achilles_rupture_binary = if_else(injury_type == "rupture", 1L, 0L)
  ) %>%
  filter(achilles_rupture_binary == 1L) %>%
  arrange(player_name, injury_date) %>%
  distinct(player_name, .keep_all = TRUE) %>%   # keep first rupture per player
  select(player_name, injury_date, injury_desc,
         injury_type, injury_severity, achilles_rupture_binary)

# Merge historical + 2024–25 manual cases
achilles_ruptures_FULL <-
  bind_rows(achilles_ruptures_HISTORICAL, new_achilles_cases_2425) %>%
  arrange(injury_date)

write_csv(achilles_ruptures_FULL, "achilles_ruptures_recency_COMPARISON.csv")

glue("Confirmed Achilles ruptures in dataset (1990–2025): \\
      {n_distinct(achilles_ruptures_FULL$player_name)} unique players")

## 4b. Clean box scores -----------------------------------------------------
box_scores_clean <-
  all_seasons_box %>%
  rename(player_name = athlete_display_name) %>%
  mutate(
    player_name = str_squish(player_name),
    game_date   = as.Date(game_date)
  ) %>%
  filter(season_type == 2)
## 4c. Game-level feature engineering --------------------------------------
# Rolling load, rest, B2B — shared across Part 1 and Part 2.
# These are the features that connect the causal and predictive stories.

create_game_features <- function(df) {
  df %>%
    arrange(player_name, game_date) %>%
    group_by(player_name) %>%
    mutate(
      # Lagged 3-game rolling average minutes — captures recent workload intensity
      # Lagged so it represents load BEFORE the current game (no data leakage)
      rolling_minutes_3    = lag(
        zoo::rollapplyr(minutes, width = 3, FUN = mean,
                        fill = NA, partial = TRUE)
      ),
      days_since_last_game = as.numeric(game_date - lag(game_date)),
      # Back-to-back: less than or equal to 1 rest day
      back_to_back         = if_else(days_since_last_game <= 1, 1L, 0L,
                                     missing = 0L),
      days_rest            = coalesce(
        days_since_last_game,
        median(days_since_last_game, na.rm = TRUE)
      ),
      home_game            = if_else(home_away == "home", 1L, 0L),
      # Fill NAs on first games of career
      rolling_minutes_3    = replace_na(rolling_minutes_3, 0),
      days_rest            = replace_na(days_rest,
                                        median(days_rest, na.rm = TRUE))
    ) %>%
    ungroup()
}

box_with_features <-
  box_scores_clean %>%
  select(game_id, game_date, season, player_name, athlete_id,
         team_id, home_away, minutes, starter,
         points, rebounds, assists, steals, blocks, turnovers, fouls) %>%
  create_game_features()

## 4d. Player-season aggregates --------------------------------------------
# Single aggregation used by both parts.

nba_player_season_AGG <-
  box_with_features %>%
  group_by(player_name, athlete_id, season) %>%
  summarise(
    avg_rolling_minutes_3 = mean(rolling_minutes_3, na.rm = TRUE),
    total_minutes         = sum(minutes,   na.rm = TRUE),
    back_to_back_games    = sum(back_to_back, na.rm = TRUE),
    avg_days_rest         = mean(days_rest, na.rm = TRUE),
    total_points          = sum(points,    na.rm = TRUE),
    total_rebounds        = sum(rebounds,  na.rm = TRUE),
    total_assists         = sum(assists,   na.rm = TRUE),
    total_steals          = sum(steals,    na.rm = TRUE),
    total_blocks          = sum(blocks,    na.rm = TRUE),
    total_turnovers       = sum(turnovers, na.rm = TRUE),
    total_fouls           = sum(fouls,     na.rm = TRUE),
    home_game_pct         = mean(home_game, na.rm = TRUE),
    games_played          = n(),
    .groups               = "drop"
  )

## 4e. Ever-injured flag ---------------------------------------------------
# Any player who ever appeared in the injury log (all injury types).

ever_injured_flags <-
  bind_rows(
    injuries_raw %>%
      rename(player_name_1 = Relinquished, player_name_2 = Acquired) %>%
      mutate(
        player_name = coalesce(player_name_1, player_name_2),
        player_name = str_squish(str_remove_all(player_name, "\\(.*\\)"))
      ) %>%
      filter(!is.na(player_name)) %>%
      distinct(player_name),
    espn_injury_CLEAN %>%
      distinct(NAME) %>%
      rename(player_name = NAME)
  ) %>%
  distinct(player_name) %>%
  mutate(ever_injured = 1L)

###==========================================================================###
###  5. CHECKPOINT                                                             ###
###==========================================================================###

save.image("00_Shared_Data.RData")

message("✓ 00_Data_Setup.R complete — shared objects ready:")
message("  → achilles_ruptures_FULL       : confirmed rupture cases 1990–2025")
message("  → box_scores_clean             : game-level NBA box scores 2002–2025")
message("  → box_with_features            : box scores + load/rest features")
message("  → nba_player_season_AGG        : player-season aggregates")
message("  → ever_injured_flags           : any prior injury flag")
message("  → theme_nba()                  : custom ggplot theme")
message("")
message("  Source this file first, then run Part 1 or Part 2.")
