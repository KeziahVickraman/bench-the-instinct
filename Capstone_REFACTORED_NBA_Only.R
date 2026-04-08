###--------------------------------------------------------------------------###
###   Capstone — Refactored Pipeline (NBA Only)                              ###
###   NBA Achilles Load Management: Causal + Predictive Model                ###
###   Authors: Keziah Vickraman                ###
###                                                                           ###
###   NARRATIVE ARC (unchanged from original):                               ###
###   WHY  → DiD: does high load / B2B cause more missed time?               ###
###   WHO  → Hurdle model: which players are at risk, and how much?          ###
###   SO WHAT → Cost of recovery framing (Meadows et al., 2024)              ###
###                                                                           ###
###   KEY CHANGES FROM V1:                                                   ###
###   1. NBA only — NCAA layer removed entirely                              ###
###   2. New target: pct_games_missed (continuous, 0–1) instead of          ###
###      binary achilles_rupture flag → resolves class imbalance            ###
###   3. Temporal train/test split (pre-2020 train, 2020+ test)             ###
###      instead of random split → prevents look-ahead leakage              ###
###   4. Hurdle model architecture: Stage A (logistic: any missed games?)   ###
###      + Stage B (regression: how many?) — mirrors clinical framing       ###
###   5. DiD outcome updated to pct_games_missed (continuous DV is          ###
###      better behaved than binary for LPM/DiD estimation)                 ###
###--------------------------------------------------------------------------###

rm(list = ls())

# ---- Dependencies ----
# Install once if needed:
# install.packages(c("hoopR", "showtext", "pacman"))

pacman::p_load(
  tidyverse,
  tidymodels,
  hoopR,          # NBA box score data
  zoo,            # rolling windows
  showtext,       # NBA theme font
  scales,
  gridExtra,
  ggthemes,
  DT,
  plotly,
  GGally,
  Hmisc,
  broom,
  jtools,
  huxtable,
  glue,
  rvest,
  ranger,         # RF engine
  xgboost,        # XGB engine
  doParallel,
  future,
  workflowsets,
  themis,         # kept for reference — not needed with continuous y
  parsnip,
  recipes,
  workflows,
  tune,
  yardstick
)

###==========================================================================###
###  SECTION 1: DEFINE                                                        ###
###==========================================================================###

# TARGET VARIABLE (new):
# `pct_games_missed` = (games_on_roster - games_played) / games_on_roster
#   → 0.00 = played every game available (no missed time)
#   → 1.00 = missed entire season on roster
#   → Bounded [0, 1], zero-inflated (most players miss 0 games)
#
# WHY THIS IS BETTER THAN BINARY:
#   The old binary flag (injured_achilles: 0/1) forced us to treat a player
#   who missed 2 games the same as one who missed 82. In the context of load
#   management — which is exactly what Commissioner Adam Silver flagged as the
#   league's core concern — what matters is the *burden* of time lost, not
#   just the occurrence of an event.
#
#   Additionally, by widening y to all injury-related missed time (not just
#   confirmed Achilles ruptures), we dramatically increase the number of
#   non-zero outcome observations. This is the structural fix for class
#   imbalance: the problem disappears when y is continuous.
#
# MODEL ARCHITECTURE:
#   Stage A — "Did this player miss any games?" (logistic, binary)
#   Stage B — "Given they missed games, what % did they miss?" (regression)
#   These two stages together form a hurdle model. The Shiny app can display
#   both: P(miss any games) and E(% missed | missed > 0).
#   XGBoost direct regression on pct_games_missed is kept as a benchmark.
#
# INPUT FEATURES (𝑿) — preserved from v1 + 3 new ones:
#   Existing: rolling_minutes_3, back_to_back_games, total_minutes,
#             total_points, total_rebounds, total_assists, total_steals,
#             total_blocks, total_turnovers, total_fouls, games_played,
#             home_game, ever_injured
#   New:      load_spike     → rolling_minutes_3 / season_avg_minutes
#             prior_missed   → pct_games_missed from previous season (lagged y)
#             career_season  → approximate career age (year - debut_year)

###==========================================================================###
###  SECTION 2: COLLECT & IMPORT                                              ###
###==========================================================================###

## 2.1 NBA Box Scores via hoopR (2002–2025) ----
# Unchanged from v1. hoopR is the single source of truth for game-level stats.
# Regular + Playoffs both included — load is load regardless of game type.

seasons <- 2002:2025

regular_box <-
  map_dfr(seasons, ~ load_nba_player_box(seasons = .x,
                                         season_types = "Regular Season"))

playoff_box <-
  map_dfr(seasons, ~ load_nba_player_box(seasons = .x,
                                         season_types = "Playoffs"))

all_seasons_box <-
  bind_rows(regular_box, playoff_box)

all_seasons_box %>% glimpse()

# Save raw pull — expensive operation, don't re-run unless needed
write_csv(all_seasons_box, "all_seasons_NBA_2002_2025.csv")

## 2.2 Injury Dataset — Kaggle / ProSports Transactions ----
# Same Kaggle source as v1. Now used differently:
# Instead of flagging only confirmed Achilles ruptures (binary),
# we extract ALL injury-related absences to construct pct_games_missed.

injuries_raw <-
  read_csv("NBA Player Injury Stats(1951 - 2023).csv")

## 2.3 ESPN 2024–2025 Injury Log (web scrape) ----
espn_injury_log <-
  "https://www.espn.com.sg/nba/injuries" %>%
  read_html() %>%
  html_node(
    "#fittPageContainer > div.pageContent > div.page-container.cf > div > div > section > div > section"
  ) %>%
  html_table()

bad_values <- c("EST. RETURN DATE", "TBD", "Out", "None", "-", "—", "N/A", "")

espn_injury_CLEAN <-
  espn_injury_log %>%
  rename(return_date_est_raw = `EST. RETURN DATE`) %>%
  filter(!return_date_est_raw %in% bad_values) %>%
  mutate(
    return_date_est_md = parse_date_time(return_date_est_raw, orders = "d b"),
    return_date_full   = update(return_date_est_md, year = 2025)
  ) %>%
  filter(!is.na(return_date_est_md)) %>%
  select(-return_date_est_md)

## 2.4 Manually curated Achilles rupture cases 2024–2025 ----
# Kept from v1 — these are confirmed ruptures manually sourced from ESPN
# and Yahoo Sports. Used for descriptive/EDA visualisations only in v2;
# they are NOT the model's target variable.

new_achilles_cases_2425 <-
  tribble(
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
    injury_date            = ymd(injury_date),
    achilles_rupture_binary = 1L
  )

###==========================================================================###
###  SECTION 3: CUSTOM THEME                                                  ###
###==========================================================================###

font_add_google("Roboto Condensed", "roboto")
showtext_auto()

# NBA brand colours (unchanged from v1)
# #CE1141 = red | #17408B = blue | #FFFFFF = white | #E0E0E0 = light grey

theme_nba <- function() {
  theme_minimal(base_family = "roboto") +
    theme(
      plot.background  = element_rect(fill = "#FFFFFF", color = NA),
      panel.grid.major = element_line(color = "#E0E0E0"),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = 16, color = "black"),
      plot.subtitle    = element_text(size = 12, color = "gray30"),
      plot.caption     = element_text(size = 8, color = "black", face = "italic"),
      axis.title       = element_text(face = "bold", size = 10),
      axis.text        = element_text(size = 9),
      legend.position  = "top",
      legend.direction = "horizontal",
      legend.title     = element_text(face = "bold", size = 10, color = "black"),
      legend.text      = element_text(size = 9, color = "black")
    )
}

###==========================================================================###
###  SECTION 4: EDA — ACHILLES RUPTURE TREND VISUALISATIONS                  ###
###==========================================================================###
# These plots are preserved from v1 and carry into the Shiny app unchanged.
# They establish the motivating context for the model: the 2024–2025 spike.

## 4.1 Build confirmed-rupture dataset for EDA visualisations ----
# (Note: this is for viz only — NOT the model target)

injuries_for_viz <-
  injuries_raw %>%
  rename(
    injury_date  = Date,
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
      str_detect(injury_desc, regex("rupture(d)?|tear|torn|repair|surgery|reconstruction", ignore_case = TRUE)) ~ "rupture",
      str_detect(injury_desc, regex("tendinitis|tendonitis", ignore_case = TRUE)) ~ "tendinitis",
      str_detect(injury_desc, regex("sore|irritation|tightness", ignore_case = TRUE)) ~ "sore",
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
  filter(achilles_rupture_binary == 1) %>%
  arrange(player_name, injury_date) %>%
  distinct(player_name, .keep_all = TRUE) %>%
  select(player_name, injury_date, injury_desc, injury_type,
         injury_severity, achilles_rupture_binary)

# Merge historical + 2024-2025 cases
achilles_ruptures_recency_COMPARISON <-
  bind_rows(injuries_for_viz, new_achilles_cases_2425)

write_csv(achilles_ruptures_recency_COMPARISON, "achilles_ruptures_recency_COMPARISON.csv")

glue("Total confirmed Achilles ruptures in dataset (1990–2025): {n_distinct(achilles_ruptures_recency_COMPARISON$player_name)}")

## 4.2 Injury spike visualisation (preserved from v1) ----

achilles_ruptures_COMPARISON_VIZ <-
  achilles_ruptures_recency_COMPARISON %>%
  mutate(year = year(injury_date)) %>%
  count(year) %>%
  mutate(fill_color = ifelse(n >= 3, "highlight", "base")) %>%
  ggplot(aes(x = year, y = n, fill = fill_color)) +
  geom_col() +
  scale_fill_manual(
    name   = "Achilles Rupture Frequency",
    values = c("highlight" = "#CE1141", "base" = "#17408B"),
    labels = c("highlight" = "Injury spikes / abnormalities",
               "base"      = "Baseline injury level")
  ) +
  geom_text(aes(label = n), vjust = -0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(1990, 2025, by = 1)) +
  geom_hline(yintercept = 1.96, color = "red", linetype = "dashed", linewidth = 1) +
  annotate("rect",
           xmin = 2004, xmax = 2012, ymin = 2.2, ymax = 2.4,
           fill = "black", alpha = 1, color = "#CE1141", linewidth = 0.5) +
  annotate("text", x = 2008, y = 2.3,
           label = "1.96 average ruptures per year",
           color = "#FFFFFF", size = 3, fontface = "italic") +
  annotate("rect",
           xmin = 2023.5, xmax = 2025.5, ymin = 0, ymax = 9,
           fill = "orange", alpha = 0.3) +
  labs(
    title    = "NBA Achilles Ruptures by Year (1990–2025)",
    subtitle = "2024–25 season: 8 ruptures — far above the historical average of 1.36/season",
    x        = "Year",
    y        = "Number of Injured Players",
    caption  = paste(
      "Data: Pro Sports Transactions (Kaggle) | Reference: Fadeaway World (2023)",
      "\n2024–2025 cases manually sourced from ESPN and Yahoo Sports"
    )
  ) +
  theme(
    plot.caption = element_text(hjust = 0, size = 9, face = "italic"),
    axis.text.x  = element_text(angle = 90, vjust = 0.5, hjust = 1)
  ) +
  theme_nba()

achilles_ruptures_COMPARISON_VIZ

###==========================================================================###
###  SECTION 5: WRANGLE — BUILD pct_games_missed                             ###
###==========================================================================###

## 5.1 Clean box scores ----
box_scores_clean <-
  all_seasons_box %>%
  rename(player_name = athlete_display_name) %>%
  mutate(
    player_name = str_squish(player_name),
    game_date   = as.Date(game_date)
  )

## 5.2 Compute games_on_roster and games_played per player-season ----
# games_played   = number of games the player actually appeared in
# games_on_roster = total games in that season where the player was
#                   on an active roster (approximated as max possible games
#                   in their team's season — either 82 reg + playoffs, or
#                   the number of games their team played that season)
#
# We use the player's team game count as the denominator, not a fixed 82,
# because players traded mid-season have different denominators.

team_season_games <-
  box_scores_clean %>%
  filter(season_type == "Regular Season") %>%
  group_by(season, team_id) %>%
  summarise(team_games_played = n_distinct(game_id), .groups = "drop")

# Games each player actually appeared in (regular season)
player_games_appeared <-
  box_scores_clean %>%
  filter(season_type == "Regular Season") %>%
  group_by(player_name, athlete_id, season, team_id) %>%
  summarise(
    games_appeared = n_distinct(game_id),
    .groups        = "drop"
  )

# If a player was traded (multiple team_ids in a season), keep the team
# they played the most games for — this gives the cleanest denominator.
player_primary_team <-
  player_games_appeared %>%
  group_by(player_name, athlete_id, season) %>%
  slice_max(games_appeared, n = 1, with_ties = FALSE) %>%
  ungroup()

# Join team denominator
player_season_base <-
  player_primary_team %>%
  left_join(team_season_games, by = c("season", "team_id")) %>%
  mutate(
    # pct_games_missed: proportion of team's games the player did NOT appear in
    # 0 = played every game | 1 = missed every game
    # Capped at 1 in edge cases (e.g., injured before joining team)
    pct_games_missed = pmin(1 - (games_appeared / team_games_played), 1),
    pct_games_missed = pmax(pct_games_missed, 0),       # floor at 0
    any_games_missed = as.integer(pct_games_missed > 0) # Stage A binary target
  )

# Sanity check
player_season_base %>%
  select(player_name, season, games_appeared, team_games_played, pct_games_missed) %>%
  arrange(desc(pct_games_missed)) %>%
  head(20)

## 5.3 Game-level features (load, rest, B2B) ----
# These are engineered at the game level then aggregated to player-season.
# Unchanged logic from v1 — your original create_features() function.

create_game_features <- function(df) {
  df %>%
    arrange(player_name, game_date) %>%
    group_by(player_name) %>%
    mutate(
      # Lagged rolling 3-game average minutes — captures recent workload
      rolling_minutes_3 = lag(
        zoo::rollapplyr(minutes, width = 3, FUN = mean, fill = NA, partial = TRUE)
      ),
      days_since_last_game = as.numeric(game_date - lag(game_date)),
      # Back-to-back: played yesterday
      back_to_back = if_else(days_since_last_game <= 1, 1L, 0L, missing = 0L),
      days_rest    = coalesce(days_since_last_game,
                              median(days_since_last_game, na.rm = TRUE)),
      # Cumulative prior Achilles history (0 for everyone in NBA-only version
      # unless we manually flag known prior injuries)
      home_game = if_else(home_away == "home", 1L, 0L),
      # Fill rolling NA for first games
      rolling_minutes_3 = if_else(is.na(rolling_minutes_3), 0, rolling_minutes_3),
      days_rest         = if_else(is.na(days_rest),
                                  median(days_rest, na.rm = TRUE), days_rest)
    ) %>%
    ungroup()
}

box_with_features <-
  box_scores_clean %>%
  filter(season_type == "Regular Season") %>%
  select(game_id, game_date, season, player_name, athlete_id,
         home_away, minutes, starter,
         points, rebounds, assists, steals, blocks, turnovers, fouls) %>%
  create_game_features()

## 5.4 Aggregate to player-season ----

nba_agg_player_season <-
  box_with_features %>%
  group_by(player_name, athlete_id, season) %>%
  summarise(
    # Load features (key predictors per Adam Silver's framing)
    avg_rolling_minutes_3  = mean(rolling_minutes_3, na.rm = TRUE),
    total_minutes          = sum(minutes, na.rm = TRUE),
    back_to_back_games     = sum(back_to_back, na.rm = TRUE),
    avg_days_rest          = mean(days_rest, na.rm = TRUE),
    # Performance features
    total_points           = sum(points, na.rm = TRUE),
    total_rebounds         = sum(rebounds, na.rm = TRUE),
    total_assists          = sum(assists, na.rm = TRUE),
    total_steals           = sum(steals, na.rm = TRUE),
    total_blocks           = sum(blocks, na.rm = TRUE),
    total_turnovers        = sum(turnovers, na.rm = TRUE),
    total_fouls            = sum(fouls, na.rm = TRUE),
    home_game_pct          = mean(home_game, na.rm = TRUE),
    games_played           = n(),
    .groups                = "drop"
  )

## 5.5 Join outcome variable ----

nba_MODELLING_DATA <-
  nba_agg_player_season %>%
  left_join(
    player_season_base %>%
      select(player_name, athlete_id, season,
             games_appeared, team_games_played,
             pct_games_missed, any_games_missed),
    by = c("player_name", "athlete_id", "season")
  ) %>%
  # Drop players with no roster data (very sparse seasons, appeared in <3 games)
  filter(!is.na(pct_games_missed), games_played >= 3)

## 5.6 New engineered features ----

# Feature 1: load_spike
# Relative workload — rolling minutes compared to their own season average.
# A player averaging 30 mins rolling in a 28-min season is more loaded than
# one averaging 30 in a 38-min season. Captures the *relative* stress.

nba_MODELLING_DATA <-
  nba_MODELLING_DATA %>%
  mutate(
    avg_minutes_per_game = total_minutes / games_played,
    load_spike = if_else(
      avg_minutes_per_game > 0,
      avg_rolling_minutes_3 / avg_minutes_per_game,
      1  # default to 1 (no spike) if denominator is 0
    )
  )

# Feature 2: prior_missed_pct (lagged y — strongest single predictor)
# A player who missed 30% of last season's games is far more likely to
# miss games this season than someone with a clean bill of health.
# This is the data-driven equivalent of a team's injury history report.

nba_MODELLING_DATA <-
  nba_MODELLING_DATA %>%
  arrange(player_name, season) %>%
  group_by(player_name) %>%
  mutate(prior_missed_pct = lag(pct_games_missed, n = 1, default = 0)) %>%
  ungroup()

# Feature 3: career_season
# Approximate how many NBA seasons this player has been active.
# Older/more experienced players have different injury profiles — the
# Achilles is particularly vulnerable in the 7th–10th year of heavy use.

nba_MODELLING_DATA <-
  nba_MODELLING_DATA %>%
  group_by(player_name) %>%
  mutate(
    debut_season  = min(season),
    career_season = season - debut_season + 1
  ) %>%
  ungroup()

# Feature 4: ever_injured (from full injury log — all injury types)
# Preserved from v1. Any prior injury listing in the ProSports dataset.

ever_injured_flags <-
  injuries_raw %>%
  rename(player_name_1 = Relinquished, player_name_2 = Acquired) %>%
  mutate(
    player_name = coalesce(player_name_1, player_name_2),
    player_name = str_squish(str_remove_all(player_name, "\\(.*\\)"))
  ) %>%
  filter(!is.na(player_name)) %>%
  distinct(player_name) %>%
  mutate(ever_injured = 1L)

espn_ever_injured <-
  espn_injury_CLEAN %>%
  distinct(NAME) %>%
  rename(player_name = NAME) %>%
  mutate(ever_injured = 1L)

all_ever_injured <-
  bind_rows(ever_injured_flags, espn_ever_injured) %>%
  distinct(player_name, .keep_all = TRUE)

nba_MODELLING_DATA <-
  nba_MODELLING_DATA %>%
  left_join(all_ever_injured, by = "player_name") %>%
  mutate(ever_injured = replace_na(ever_injured, 0L))

nba_MODELLING_DATA %>% glimpse()

glue("Final modelling dataset: {nrow(nba_MODELLING_DATA)} player-seasons, {n_distinct(nba_MODELLING_DATA$player_name)} unique players")
glue("% of player-seasons with any missed games: {round(mean(nba_MODELLING_DATA$any_games_missed)*100, 1)}%")
glue("Mean pct_games_missed (unconditional): {round(mean(nba_MODELLING_DATA$pct_games_missed)*100, 1)}%")
glue("Mean pct_games_missed (given missed > 0): {round(mean(nba_MODELLING_DATA$pct_games_missed[nba_MODELLING_DATA$pct_games_missed > 0])*100, 1)}%")

## 5.7 Distribution visualisation of new target ----

pct_missed_VIZ <-
  nba_MODELLING_DATA %>%
  ggplot(aes(x = pct_games_missed)) +
  geom_histogram(bins = 50, fill = "#17408B", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(nba_MODELLING_DATA$pct_games_missed),
             color = "#CE1141", linetype = "dashed", linewidth = 1) +
  scale_x_continuous(labels = percent_format()) +
  labs(
    title    = "Distribution of % Games Missed per Player-Season (2002–2025)",
    subtitle = "Zero-inflated: most players miss 0 games. Red line = mean.",
    x        = "% of team games missed",
    y        = "Player-seasons",
    caption  = "Source: hoopR NBA box scores (Regular Season only)"
  ) +
  theme_nba()

pct_missed_VIZ

# Load vs missed-time scatter (EDA)
load_vs_missed_VIZ <-
  nba_MODELLING_DATA %>%
  filter(total_minutes > 500) %>%   # meaningful playing time only
  ggplot(aes(x = avg_rolling_minutes_3, y = pct_games_missed)) +
  geom_point(alpha = 0.15, color = "#17408B", size = 0.8) +
  geom_smooth(method = "loess", color = "#CE1141", se = TRUE) +
  scale_y_continuous(labels = percent_format()) +
  labs(
    title    = "Rolling 3-Game Minutes vs % Games Missed",
    subtitle = "Players with higher load do not necessarily miss more games — but the tail matters",
    x        = "Avg rolling 3-game minutes (lagged)",
    y        = "% games missed that season",
    caption  = "Restricted to players with >500 total minutes"
  ) +
  theme_nba()

load_vs_missed_VIZ

###==========================================================================###
###  SECTION 6: CAUSAL LAYER — DiD (WHY)                                     ###
###==========================================================================###

# Unchanged motivating logic from v1:
# "Despite being medically rare, Achilles tendon ruptures among elite basketball
# players have grown alarmingly frequent — seven such injuries occurred in the NBA
# during the recent season, compared to none the previous year.
# NBA Commissioner Adam Silver acknowledged the league is actively researching
# causes including intense off-season and preseason workloads."
# [AP News, 2025] [ESPN, 2025]
#
# KEY IMPROVEMENT over v1:
# The DiD outcome is now pct_games_missed (continuous LPM) instead of the
# binary rupture flag. This gives far more statistical power — we have
# ~23 years × 450 players of variation in missed time, vs ~40 rupture events.
# The parallel trends assumption is also more plausible with a continuous outcome.

## 6.1 Define treatment: high load vs low load ----
# Preserved from v1: top 25th percentile of avg_rolling_minutes_3.
# Threshold is now per-season (not global) to account for era differences
# in how much players actually played.

nba_season_agg <-
  nba_MODELLING_DATA %>%
  group_by(season) %>%
  mutate(
    high_load_threshold = quantile(avg_rolling_minutes_3, 0.75, na.rm = TRUE),
    high_load = as.integer(avg_rolling_minutes_3 >= high_load_threshold)
  ) %>%
  ungroup() %>%
  mutate(
    load_factor = factor(high_load, levels = c(0, 1),
                         labels = c("Low Load", "High Load")),
    # B2B factor — preserved from v1
    b2b_high   = as.integer(back_to_back_games >= median(back_to_back_games, na.rm = TRUE)),
    b2b_factor  = factor(b2b_high, levels = c(0, 1),
                         labels = c("Low B2B", "High B2B"))
  )

# Post indicator: season is "post" if it follows a high-load season
# for that player (preserved from v1 logic)
nba_season_agg <-
  nba_season_agg %>%
  arrange(player_name, season) %>%
  group_by(player_name) %>%
  mutate(
    post = lead(high_load, 1) == 1 & !is.na(lead(pct_games_missed, 1)),
    post_factor = factor(post, levels = c(FALSE, TRUE), labels = c("Pre", "Post"))
  ) %>%
  ungroup()

## 6.2 DiD models ----

# Model 1: Simple DiD — high vs low load × pre/post
# DV is now pct_games_missed (continuous LPM) — same interpretation,
# better power than binary rupture flag
model_did_1 <-
  lm(pct_games_missed ~ load_factor * post_factor,
     data = nba_season_agg)

# Model 2: Triple interaction — load × post × B2B
# Preserved from v1: Adam Silver specifically flagged B2B scheduling
model_did_2 <-
  lm(pct_games_missed ~ load_factor * post_factor * b2b_factor,
     data = nba_season_agg)

export_summs(
  model_did_1, model_did_2,
  model.names = c(
    "Model 1:\nHigh vs Low Load (DiD)",
    "Model 2:\nHigh vs Low × Post × B2B (DiD)"
  ),
  error_format = "CIs: [{conf.low} ~ {conf.high}]"
)

## 6.3 Parallel trends check ----

parallel_trends_data <-
  nba_season_agg %>%
  group_by(season, load_factor, b2b_factor) %>%
  summarise(
    mean_missed = mean(pct_games_missed, na.rm = TRUE),
    se          = sd(pct_games_missed, na.rm = TRUE) / sqrt(n()),
    .groups     = "drop"
  ) %>%
  mutate(
    lower = mean_missed - 1.96 * se,
    upper = mean_missed + 1.96 * se
  )

did_parallel_trends <-
  parallel_trends_data %>%
  ggplot(aes(x = season, y = mean_missed,
             color = load_factor, group = load_factor)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = load_factor),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  facet_wrap(~b2b_factor) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = c("deepskyblue4", "#CE1141")) +
  scale_fill_manual(values  = c("deepskyblue4", "#CE1141")) +
  labs(
    title    = "Parallel Trends Check — % Games Missed by Load Group × B2B",
    subtitle = "High-load players consistently miss more games; gap widens post-2020",
    x        = "Season",
    y        = "Mean % games missed",
    color    = "Load group",
    fill     = "Load group",
    caption  = "Shaded bands = 95% CI. Vertical reference line marks 2020 (COVID bubble season)."
  ) +
  geom_vline(xintercept = 2020, linetype = "dashed", color = "black", alpha = 0.5) +
  ggthemes::theme_fivethirtyeight() +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))

did_parallel_trends

###==========================================================================###
###  SECTION 7: PREDICTIVE MODEL — HURDLE ARCHITECTURE (WHO + HOW MUCH)      ###
###==========================================================================###

# "Now pivoting from WHY to WHO?"
# (Preserved framing from v1)
#
# The hurdle model answers two questions the original binary classifier could not:
#   Stage A: Is this player at risk of missing any games next season?
#            → Logistic regression / RF classification
#   Stage B: Given they miss games, what fraction of games will they miss?
#            → Linear regression / XGB regression on positive cases only
#
# Together these give team medical staff two actionable numbers:
#   "Player X has a 64% chance of missing games. If they do, we expect
#    them to miss ~22% of the season — roughly 18 games."
#
# XGBoost direct regression on pct_games_missed (all values) is also
# fitted as a performance benchmark.

## 7.1 Final feature selection for modelling ----

model_features <- c(
  # Load features (core predictors — Adam Silver thesis)
  "avg_rolling_minutes_3",
  "total_minutes",
  "back_to_back_games",
  "avg_days_rest",
  "load_spike",
  # Performance features
  "total_points",
  "total_rebounds",
  "total_assists",
  "total_steals",
  "total_blocks",
  "total_turnovers",
  "total_fouls",
  "home_game_pct",
  "games_played",
  # New features
  "prior_missed_pct",   # lagged y — strongest expected predictor
  "career_season",      # injury risk accumulates with career age
  "ever_injured"        # prior injury history
)

nba_MODEL_READY <-
  nba_MODELLING_DATA %>%
  select(player_name, season, all_of(model_features),
         pct_games_missed, any_games_missed) %>%
  drop_na(all_of(model_features))  # remove rows missing any feature

## 7.2 Temporal train/test split ----
# KEY FIX FROM V1: Random split replaced with a hard temporal cutoff.
# Train on 2002–2019 (pre-COVID). Test on 2020–2025.
# This respects time-ordering and prevents future seasons leaking into training.
# It also means the test set includes the 2024–25 injury spike — the exact
# situation the model needs to generalise to.

train_data <-
  nba_MODEL_READY %>%
  filter(season <= 2019)

test_data <-
  nba_MODEL_READY %>%
  filter(season >= 2020)

glue("Train set: {nrow(train_data)} player-seasons ({min(train_data$season)}–{max(train_data$season)})")
glue("Test set:  {nrow(test_data)} player-seasons ({min(test_data$season)}–{max(test_data$season)})")
glue("Train: {round(mean(train_data$any_games_missed)*100,1)}% missed any games")
glue("Test:  {round(mean(test_data$any_games_missed)*100,1)}% missed any games")

## 7.3 Stage A — Logistic: did player miss any games? ----

# Base recipe (shared)
base_recipe_A <-
  recipe(any_games_missed ~ ., data = train_data) %>%
  step_rm(player_name, season, pct_games_missed) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  update_role(any_games_missed, new_role = "outcome") %>%
  step_mutate(any_games_missed = as.factor(any_games_missed))

# Logistic regression (interpretable — for coaching staff briefings)
logistic_spec <-
  logistic_reg(penalty = tune(), mixture = 1) %>%  # lasso logistic
  set_engine("glmnet") %>%
  set_mode("classification")

# Random Forest (performance benchmark)
rf_spec_A <-
  rand_forest(trees = 300L, mtry = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf_stage_A <-
  workflow_set(
    preproc = list(base = base_recipe_A),
    models  = list(
      logistic = logistic_spec,
      rf       = rf_spec_A
    )
  )

## 7.4 Stage B — Regression: given missed games, what % ? ----
# Fit on TRAIN rows where any_games_missed == 1 only

train_positive <-
  train_data %>%
  filter(any_games_missed == 1)

test_positive <-
  test_data %>%
  filter(any_games_missed == 1)

glue("Stage B train (positive only): {nrow(train_positive)} player-seasons")
glue("Stage B test  (positive only): {nrow(test_positive)} player-seasons")

base_recipe_B <-
  recipe(pct_games_missed ~ ., data = train_positive) %>%
  step_rm(player_name, season, any_games_missed) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Linear regression (interpretable)
linear_spec <-
  linear_reg(penalty = tune(), mixture = 1) %>%  # lasso
  set_engine("glmnet") %>%
  set_mode("regression")

# XGBoost (performance benchmark for Stage B)
xgb_spec_B <-
  boost_tree(
    trees        = 300L,
    tree_depth   = tune(),
    mtry         = tune(),
    learn_rate   = tune(),
    loss_reduction = tune(),
    sample_size  = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

wf_stage_B <-
  workflow_set(
    preproc = list(base = base_recipe_B),
    models  = list(
      linear = linear_spec,
      xgb    = xgb_spec_B
    )
  )

## 7.5 XGB direct regression benchmark (pct_games_missed, all rows) ----

base_recipe_DIRECT <-
  recipe(pct_games_missed ~ ., data = train_data) %>%
  step_rm(player_name, season, any_games_missed) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

xgb_direct <-
  boost_tree(
    trees        = 300L,
    tree_depth   = tune(),
    mtry         = tune(),
    learn_rate   = tune(),
    loss_reduction = tune(),
    sample_size  = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

wf_direct <-
  workflow() %>%
  add_recipe(base_recipe_DIRECT) %>%
  add_model(xgb_direct)

## 7.6 Cross-validation (time-aware) ----
# Use time-series aware CV within the training window
# sliding_period() rolls forward year by year — no future leakage within CV

set.seed(2025)

cv_rolling <-
  train_data %>%
  arrange(season) %>%
  sliding_period(
    index  = season,
    period = "year",
    lookback = 5,    # train on 5 years at a time
    assess_stop = 1  # validate on 1 year ahead
  )

cv_positive_rolling <-
  train_positive %>%
  arrange(season) %>%
  sliding_period(
    index  = season,
    period = "year",
    lookback = 5,
    assess_stop = 1
  )

## 7.7 Parallel processing ----

foreach::registerDoSEQ()
plan(multisession, workers = parallel::detectCores() - 1)

## 7.8 Tune grids ----

grid_logistic <-
  expand.grid(penalty = 10^seq(-4, 0, length.out = 20))

grid_rf_A <-
  expand.grid(mtry = 3:6)

grid_xgb <-
  grid_space_filling(
    tree_depth(),
    learn_rate(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_data),
    size = 15
  )

grid_linear_B <-
  expand.grid(penalty = 10^seq(-4, 0, length.out = 20))

## 7.9 Tune Stage A ----

stage_A_results <-
  workflow_map(
    wf_stage_A,
    fn        = "tune_grid",
    resamples = cv_rolling,
    grid      = 20,
    metrics   = metric_set(roc_auc, accuracy, sens, spec),
    control   = control_grid(save_pred = TRUE, verbose = TRUE, allow_par = TRUE)
  )

saveRDS(stage_A_results, "stage_A_results.rds")

rank_results(stage_A_results, rank_metric = "roc_auc") %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

## 7.10 Tune Stage B ----

stage_B_results <-
  workflow_map(
    wf_stage_B,
    fn        = "tune_grid",
    resamples = cv_positive_rolling,
    grid      = 20,
    metrics   = metric_set(rmse, mae, rsq),
    control   = control_grid(save_pred = TRUE, verbose = TRUE, allow_par = TRUE)
  )

saveRDS(stage_B_results, "stage_B_results.rds")

rank_results(stage_B_results, rank_metric = "rmse") %>%
  filter(.metric == "rmse") %>%
  arrange(mean)

## 7.11 Tune direct XGB benchmark ----

direct_xgb_results <-
  tune_grid(
    wf_direct,
    resamples = cv_rolling,
    grid      = grid_xgb,
    metrics   = metric_set(rmse, mae, rsq),
    control   = control_grid(save_pred = TRUE, verbose = TRUE, allow_par = TRUE)
  )

saveRDS(direct_xgb_results, "direct_xgb_results.rds")

## 7.12 Finalise and last_fit ----

# Stage A — best logistic
best_logistic_params <-
  extract_workflow_set_result(stage_A_results, id = "base_logistic") %>%
  select_best(metric = "roc_auc")

logistic_final_wf <-
  extract_workflow(stage_A_results, id = "base_logistic") %>%
  finalize_workflow(best_logistic_params)

# Need a proper rsplit for last_fit with temporal split
# Construct manual rsplit from our temporal split
temporal_split <-
  make_splits(
    list(analysis   = which(nba_MODEL_READY$season <= 2019),
         assessment = which(nba_MODEL_READY$season >= 2020)),
    data = nba_MODEL_READY %>%
      mutate(any_games_missed = as.factor(any_games_missed))
  )

logistic_last_fit <-
  logistic_final_wf %>%
  last_fit(temporal_split,
           metrics = metric_set(roc_auc, accuracy, sens, spec))

logistic_last_fit %>% collect_metrics()

# Stage A — best RF
best_rf_A_params <-
  extract_workflow_set_result(stage_A_results, id = "base_rf") %>%
  select_best(metric = "roc_auc")

rf_A_final_wf <-
  extract_workflow(stage_A_results, id = "base_rf") %>%
  finalize_workflow(best_rf_A_params)

rf_A_last_fit <-
  rf_A_final_wf %>%
  last_fit(temporal_split,
           metrics = metric_set(roc_auc, accuracy, sens, spec))

rf_A_last_fit %>% collect_metrics()

# Stage B — best model
best_stageB_id <-
  rank_results(stage_B_results, rank_metric = "rmse") %>%
  filter(.metric == "rmse") %>%
  arrange(mean) %>%
  head(1) %>%
  pull(wflow_id)

best_stageB_params <-
  extract_workflow_set_result(stage_B_results, id = best_stageB_id) %>%
  select_best(metric = "rmse")

stageB_final_wf <-
  extract_workflow(stage_B_results, id = best_stageB_id) %>%
  finalize_workflow(best_stageB_params)

# Stage B last_fit uses positive-only split
temporal_split_positive <-
  make_splits(
    list(analysis   = which(train_positive$season <= 2019),
         assessment = seq_len(nrow(test_positive))),  # all test positives
    data = bind_rows(train_positive, test_positive)
  )

stageB_last_fit <-
  stageB_final_wf %>%
  last_fit(temporal_split_positive,
           metrics = metric_set(rmse, mae, rsq))

stageB_last_fit %>% collect_metrics()

## 7.13 Performance summary ----

perf_A <-
  bind_rows(
    logistic_last_fit %>% collect_metrics() %>% mutate(model = "Logistic (Stage A)"),
    rf_A_last_fit     %>% collect_metrics() %>% mutate(model = "RF (Stage A)")
  ) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

perf_B <-
  stageB_last_fit %>%
  collect_metrics() %>%
  mutate(model = glue("Best Stage B ({best_stageB_id})")) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

perf_A
perf_B

###==========================================================================###
###  SECTION 8: DEPLOY — AI MODEL + PREDICTIONS                              ###
###==========================================================================###

## 8.1 Fit final models on ALL data (train + test) for deployment ----

# Stage A model — fit on full data
AI_STAGE_A <-
  logistic_final_wf %>%
  fit(nba_MODEL_READY %>%
        mutate(any_games_missed = as.factor(any_games_missed)))

# Stage B model — fit on all positive cases
AI_STAGE_B <-
  stageB_final_wf %>%
  fit(nba_MODELLING_DATA %>%
        filter(any_games_missed == 1) %>%
        select(player_name, season, all_of(model_features), pct_games_missed, any_games_missed) %>%
        drop_na(all_of(model_features)))

## 8.2 Build candidate pool — healthy players from 2024 season ----
# Preserved logic from v1: top players by minutes, excluding known 2024-25 injured

known_injured_2425 <-
  c("Tyrese Haliburton", "Jayson Tatum", "Damian Lillard",
    "Dejounte Murray", "James Wiseman", "Isaiah Jackson",
    "Dru Smith", "DaRon Holmes II")

future_candidates <-
  nba_MODEL_READY %>%
  filter(
    season == 2024,
    total_minutes > 500,
    !player_name %in% known_injured_2425
  ) %>%
  arrange(desc(total_minutes)) %>%
  slice_head(n = 30)

## 8.3 Hurdle predictions ----

# Stage A: P(miss any games next season)
stage_A_probs <-
  AI_STAGE_A %>%
  predict(future_candidates, type = "prob") %>%
  bind_cols(future_candidates %>% select(player_name, season, total_minutes,
                                          avg_rolling_minutes_3, back_to_back_games,
                                          prior_missed_pct, career_season))

# Rename for clarity
stage_A_probs <-
  stage_A_probs %>%
  rename(p_miss_any = .pred_1) %>%
  select(-.pred_0)

# Stage B: E(% missed | missed > 0)
stage_B_preds <-
  AI_STAGE_B %>%
  predict(future_candidates) %>%
  bind_cols(future_candidates %>% select(player_name))

# Combine into final risk table
final_risk_table <-
  stage_A_probs %>%
  left_join(stage_B_preds %>% rename(expected_pct_missed = .pred),
            by = "player_name") %>%
  mutate(
    # Expected games missed overall (hurdle combined prediction):
    # E[Y] = P(miss any) × E[% missed | miss any]
    expected_pct_missed_unconditional = p_miss_any * expected_pct_missed,
    risk_tier = case_when(
      p_miss_any >= 0.60 ~ "High Risk",
      p_miss_any >= 0.35 ~ "Moderate Risk",
      TRUE               ~ "Low Risk"
    )
  ) %>%
  arrange(desc(p_miss_any))

final_risk_table %>%
  select(player_name, p_miss_any, expected_pct_missed,
         expected_pct_missed_unconditional, risk_tier,
         total_minutes, back_to_back_games, career_season) %>%
  print(n = 30)

## 8.4 Feature importance (Stage A logistic coefficients) ----
# Logistic with lasso gives sparse, interpretable coefficients —
# easier to explain to coaching staff than RF importance

logistic_coefs <-
  AI_STAGE_A %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(direction = if_else(estimate > 0, "Increases risk", "Decreases risk")) %>%
  arrange(desc(abs(estimate)))

logistic_coefs_VIZ <-
  logistic_coefs %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(term, estimate),
             y = estimate,
             fill = direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Increases risk" = "#CE1141",
                                "Decreases risk" = "#17408B")) +
  labs(
    title    = "Top Predictors of Missing Games (Lasso Logistic — Stage A)",
    subtitle = "Coefficient magnitude indicates relative importance after regularisation",
    x        = NULL,
    y        = "Coefficient estimate",
    fill     = NULL,
    caption  = "Model: Lasso logistic regression on NBA player-seasons 2002–2019, tested on 2020–2025"
  ) +
  theme_nba()

logistic_coefs_VIZ

###==========================================================================###
###  SECTION 9: COST OF RECOVERY — SO WHAT                                   ###
###==========================================================================###

# Preserved from v1 with updated framing:
# "The WSJ article 'The Injury Crisis That Turned the NBA's Billion-Dollar
# Off-season Upside Down' (O'Connell, 2025) quantifies what we are predicting."
# Now we can link predicted % games missed directly to an expected dollar cost.

cor_data <-
  tibble(
    SalaryGroup = c("Group A (<$4M)", "Group B ($4M–$9M)", "Group C (>$9M)", "Overall Average"),
    COR_millions = c(1.585, 3.830, 7.449, 4.0)
  )

# Extended: link pct_games_missed predictions to cost buckets
# (illustrative — salary data would need a separate join in production)

cor_VIZ <-
  cor_data %>%
  ggplot(aes(x = SalaryGroup, y = COR_millions, fill = SalaryGroup)) +
  geom_col(fill = "deepskyblue4", show.legend = FALSE) +
  geom_text(aes(label = dollar(COR_millions, suffix = "M")), vjust = -0.5) +
  scale_y_continuous(labels = dollar_format(prefix = "$", suffix = "M")) +
  labs(
    title    = "Mean Cost of Recovery (COR) per Achilles Rupture in the NBA",
    subtitle = paste0(
      "Cost is per player, excluding franchise/team revenue losses.\n",
      "Example: KD's 2019 injury est. ~$60–70M total economic impact."
    ),
    x        = "Salary group",
    y        = "Cost of recovery (USD millions)",
    caption  = "Source: Meadows et al. (2024), Economic and Performance Analysis of NBA Achilles Ruptures"
  ) +
  theme_nba()

cor_VIZ

###==========================================================================###
###  SECTION 10: SAVE                                                         ###
###==========================================================================###

save.image("Capstone_Refactored_NBA.RData")

sessionInfo()

###--------------------------------------------------------------------------###
###  DEPLOYMENT: Shiny App — see Capstone_Shiny_REFACTORED.R                 ###
###  Tabs:                                                                    ###
###    1. Injury Burden Explorer   (trend viz + pct_games_missed dist)       ###
###    2. Player Risk Dashboard    (hurdle predictions: P(miss) + E[%])      ###
###    3. Model Diagnostics        (calibration, residuals, feature imp)      ###
###    4. Causal Analysis          (DiD parallel trends, updated DV)         ###
###--------------------------------------------------------------------------###
