###--------------------------------------------------------------------------###
###   Part2_Predictive_Hurdle.R  (Final Clean Version)                       ###
###   "So who's next?" — Predicting Load-Related Game Unavailability         ###
###                                                                           ###
###   Author: Keziah Vickraman                                                ###
###                                                                           ###
###   QUESTION: Given Part 1 established that load drives the Achilles       ###
###   rupture spike, which individual players are at risk of missing         ###
###   significant game time next season — and how much?                      ###
###                                                                           ###
###   TARGET VARIABLE: pct_games_missed                                      ###
###     = 1 - (games_appeared / team_games_played)                          ###
###     Continuous [0, 1]. Built entirely from hoopR box score data.        ###
###     The injury dataset plays no role in constructing y.                 ###
###                                                                           ###
###   MODEL: XGBoost direct regression on pct_games_missed                  ###
###     A hurdle model was initially explored but diagnostic analysis        ###
###     showed 87-94% of player-seasons have any_games_missed = 1 in        ###
###     the reliable data window (2018-2025). A single-stage regression     ###
###     is more appropriate given this distribution.                        ###
###                                                                           ###
###   SURVIVORSHIP BIAS CORRECTION:                                          ###
###     PDP analysis revealed total_minutes, back_to_back_games, and        ###
###     avg_days_rest all show NEGATIVE relationships with pct_games_missed. ###
###     Players who got hurt early accumulate less of all three.            ###
###     These features were removed. avg_rolling_minutes_3 (lagged)         ###
###     is the primary load predictor - positive monotonic relationship.    ###
###                                                                           ###
###   DATA WINDOW: 2018-2025 only                                            ###
###     hoopR coverage is incomplete before 2018.                           ###
###                                                                           ###
###   TRAIN/TEST: 2018-2021 train | 2022-2025 test                          ###
###--------------------------------------------------------------------------###

source("00_Data_Setup.R")

###==========================================================================###
###  SECTION 1: BUILD pct_games_missed                                        ###
###==========================================================================###

## 1.1 Team game counts per season ------------------------------------------

team_season_games <-
  box_scores_clean %>%
  group_by(season, team_id) %>%
  summarise(team_games = n_distinct(game_id), .groups = "drop")

## 1.2 Player appearances per season ----------------------------------------

player_appearances <-
  box_scores_clean %>%
  group_by(player_name, athlete_id, season, team_id) %>%
  summarise(games_appeared = n_distinct(game_id), .groups = "drop")

player_primary_team <-
  player_appearances %>%
  group_by(player_name, athlete_id, season) %>%
  slice_max(games_appeared, n = 1, with_ties = FALSE) %>%
  ungroup()

## 1.3 Compute pct_games_missed ---------------------------------------------

player_season_outcomes <-
  player_primary_team %>%
  left_join(team_season_games, by = c("season", "team_id")) %>%
  mutate(
    pct_games_missed = pmax(0, pmin(1, 1 - (games_appeared / team_games))),
    any_games_missed = as.integer(pct_games_missed > 0)
  )

player_season_outcomes %>%
  select(player_name, season, games_appeared, team_games, pct_games_missed) %>%
  arrange(desc(pct_games_missed)) %>%
  head(10)

## 1.4 Distribution visualisation -------------------------------------------

pct_missed_dist_VIZ <-
  player_season_outcomes %>%
  ggplot(aes(x = pct_games_missed)) +
  geom_histogram(bins = 60, fill = "#17408B", color = "white", alpha = 0.85) +
  geom_vline(
    xintercept = mean(player_season_outcomes$pct_games_missed),
    color = "#CE1141", linetype = "dashed", linewidth = 1
  ) +
  annotate("text",
           x     = mean(player_season_outcomes$pct_games_missed) + 0.03,
           y     = Inf, vjust = 2,
           label = glue("Mean: {scales::percent(mean(player_season_outcomes$pct_games_missed), accuracy=0.1)}"),
           color = "#CE1141", size = 3.5, fontface = "italic") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Distribution of % Games Missed per Player-Season (2018-2025)",
    subtitle = "87-94% of player-seasons involve at least some missed games — a regression problem, not a hurdle problem.",
    x        = "% of team's games missed that season",
    y        = "Number of player-seasons",
    caption  = "Source: hoopR Regular Season box scores (2018-2025)"
  ) +
  theme_nba()

pct_missed_dist_VIZ

###==========================================================================###
###  SECTION 2: ASSEMBLE MODELLING DATASET                                    ###
###==========================================================================###

## 2.1 Join outcomes onto player-season aggregates --------------------------

nba_MODEL_READY <-
  nba_player_season_AGG %>%
  filter(games_played >= 10) %>%
  left_join(
    player_season_outcomes %>%
      select(player_name, athlete_id, season,
             games_appeared, team_games,
             pct_games_missed, any_games_missed),
    by = c("player_name", "athlete_id", "season")
  ) %>%
  filter(!is.na(pct_games_missed))

## 2.2 Feature engineering --------------------------------------------------

# load_spike: rolling minutes relative to the player's own season average
nba_MODEL_READY <-
  nba_MODEL_READY %>%
  mutate(
    avg_minutes_per_game = total_minutes / games_played,
    load_spike           = if_else(
      avg_minutes_per_game > 0,
      avg_rolling_minutes_3 / avg_minutes_per_game,
      1
    )
  )

# prior_missed_pct: lagged pct_games_missed from previous season
nba_MODEL_READY <-
  nba_MODEL_READY %>%
  arrange(player_name, season) %>%
  group_by(player_name) %>%
  mutate(prior_missed_pct = lag(pct_games_missed, n = 1, default = 0)) %>%
  ungroup()

# career_season: years active in NBA
nba_MODEL_READY <-
  nba_MODEL_READY %>%
  group_by(player_name) %>%
  mutate(
    debut_season  = min(season),
    career_season = season - debut_season + 1
  ) %>%
  ungroup()

# ever_injured: any prior appearance in the injury log
nba_MODEL_READY <-
  nba_MODEL_READY %>%
  left_join(ever_injured_flags, by = "player_name") %>%
  mutate(ever_injured = replace_na(ever_injured, 0L))

glue("Full modelling dataset: {nrow(nba_MODEL_READY)} player-seasons, \\
      {n_distinct(nba_MODEL_READY$player_name)} unique players")

## 2.3 Survivorship-corrected feature set -----------------------------------
# REMOVED: total_minutes, back_to_back_games, avg_days_rest, games_played
#   -> PDP showed negative relationships (survivorship bias)
#   -> Players who got injured early accumulate fewer of all three
# KEPT:    avg_rolling_minutes_3 (lagged) - positive monotonic
#   -> Measures load BEFORE the outcome period, escapes survivorship

model_features_CORRECTED <- c(
  "avg_rolling_minutes_3",  # primary load predictor (lagged)
  "load_spike",             # relative load vs own norm
  "prior_missed_pct",       # strongest single predictor
  "career_season",          # peak risk at seasons 4-11
  "ever_injured",           # prior injury history flag
  "total_rebounds",         # physical intensity proxy
  "total_points",
  "total_assists",
  "total_steals",
  "total_blocks",
  "total_turnovers",
  "total_fouls",
  "home_game_pct"
)

nba_MODEL_READY <-
  nba_MODEL_READY %>%
  select(player_name, season, all_of(model_features_CORRECTED),
         pct_games_missed, any_games_missed) %>%
  drop_na(all_of(model_features_CORRECTED))

###==========================================================================###
###  SECTION 3: DATA RELIABILITY CUTOFF & TRAIN/TEST SPLIT                   ###
###==========================================================================###

# hoopR coverage incomplete before 2018 — pct_games_missed unreliable pre-2018
# Part 1 DiD unaffected (uses injury CSV, not hoopR game counts)

nba_MODEL_READY_CLEAN <-
  nba_MODEL_READY %>%
  filter(season >= 2018)

train_v2 <- nba_MODEL_READY_CLEAN %>% filter(season <= 2021)
test_v2  <- nba_MODEL_READY_CLEAN %>% filter(season >= 2022)

glue("Train: {nrow(train_v2)} player-seasons (2018-2021)")
glue("Test:  {nrow(test_v2)} player-seasons (2022-2025)")
glue("Train - % with any missed games: {scales::percent(mean(train_v2$any_games_missed==1))}")
glue("Test  - % with any missed games: {scales::percent(mean(test_v2$any_games_missed==1))}")

###==========================================================================###
###  SECTION 4: RECIPE & MODEL SPECIFICATION                                  ###
###==========================================================================###

recipe_v2 <-
  recipe(pct_games_missed ~ ., data = train_v2) %>%
  step_rm(player_name, season, any_games_missed) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

xgb_spec_v2 <-
  boost_tree(
    trees          = 300L,
    tree_depth     = tune(),
    mtry           = tune(),
    learn_rate     = tune(),
    loss_reduction = tune(),
    sample_size    = tune()
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

wf_v2 <-
  workflow() %>%
  add_recipe(recipe_v2) %>%
  add_model(xgb_spec_v2)

###==========================================================================###
###  SECTION 5: TIME-AWARE CROSS-VALIDATION                                   ###
###==========================================================================###

set.seed(2025)

cv_v2 <-
  train_v2 %>%
  arrange(season) %>%
  mutate(season_date = as.Date(paste0(season, "-01-01"))) %>%
  sliding_period(
    index       = season_date,
    period      = "year",
    lookback    = 1,
    assess_stop = 1
  )

glue("CV folds: {nrow(cv_v2)}")

###==========================================================================###
###  SECTION 6: HYPERPARAMETER TUNING                                         ###
###==========================================================================###

grid_v2 <-
  grid_space_filling(
    tree_depth(),
    learn_rate(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_v2),
    size        = 15
  )

foreach::registerDoSEQ()
plan(multisession, workers = parallel::detectCores() - 1)

results_v2 <-
  tune_grid(
    wf_v2,
    resamples = cv_v2,
    grid      = grid_v2,
    metrics   = metric_set(rmse, mae, rsq),
    control   = control_grid(save_pred = TRUE,
                             verbose   = TRUE,
                             allow_par = TRUE)
  )

saveRDS(results_v2, "results_v2.rds")

collect_metrics(results_v2) %>%
  filter(.metric == "rmse") %>%
  arrange(mean) %>%
  head(5)

###==========================================================================###
###  SECTION 7: FINALISE & LAST FIT                                           ###
###==========================================================================###

best_v2_params <-
  results_v2 %>%
  select_best(metric = "rmse")

wf_v2_final <-
  wf_v2 %>%
  finalize_workflow(best_v2_params)

temporal_split_v2 <-
  make_splits(
    list(
      analysis   = which(nba_MODEL_READY_CLEAN$season <= 2021),
      assessment = which(nba_MODEL_READY_CLEAN$season >= 2022)
    ),
    data = nba_MODEL_READY_CLEAN
  )

xgb_last_fit_v2 <-
  wf_v2_final %>%
  last_fit(temporal_split_v2,
           metrics = metric_set(rmse, mae, rsq))

xgb_last_fit_v2 %>% collect_metrics()
# RMSE: 0.133 (~11 games) | MAE: 0.100 (~8 games) | R2: 0.747

###==========================================================================###
###  SECTION 8: DEPLOY — FIT ON FULL DATA                                     ###
###==========================================================================###

AI_MODEL_V2 <-
  wf_v2_final %>%
  fit(nba_MODEL_READY_CLEAN)

###==========================================================================###
###  SECTION 9: FEATURE IMPORTANCE                                            ###
###==========================================================================###

library(vip)

feature_imp_VIZ <-
  vip::vip(
    xgb_last_fit_v2 %>% extract_fit_parsnip(),
    num_features = 15,
    aesthetics   = list(fill = "#17408B", color = "white")
  ) +
  labs(
    title    = "Feature Importance — XGBoost Regression (pct_games_missed)",
    subtitle = paste(
      "Top predictors after survivorship bias correction.",
      "\navg_rolling_minutes_3 (lagged) is the primary load signal.",
      "\ntotal_minutes and back_to_back_games removed — survivorship bias confirmed via PDP."
    ),
    caption  = "Model: XGBoost V3 | Train: 2018-2021 | Test: 2022-2025"
  ) +
  theme_nba()+
  theme(axis.text.y = element_text(size = 10, color = "black"))


feature_imp_VIZ

###==========================================================================###
###  SECTION 10: MODEL VERSION COMPARISON                                     ###
###==========================================================================###

model_version_comparison <- tibble(
  version          = c("V1 - with games_played",
                       "V2 - leakage fixed",
                       "V3 - survivorship fixed (deployed)"),
  features_removed = c("none",
                       "games_played",
                       "+ total_minutes, back_to_back_games, avg_days_rest"),
  rmse             = c(0.098, 0.117, 0.133),
  mae              = c(0.061, 0.085, 0.100),
  rsq              = c(0.865, 0.806, 0.747),
  note             = c("inflated - data leakage",
                       "honest but survivorship bias remains",
                       "clean - deployed model")
)

model_version_comparison

###==========================================================================###
###  SECTION 11: WHAT-IF FUNCTION                                             ###
###==========================================================================###

whatif_predict <- function(
    model,
    avg_rolling_minutes_3 = 25,
    load_spike            = 1.0,
    prior_missed_pct      = 0,
    career_season         = 5,
    ever_injured          = 0,
    total_rebounds        = 400,
    total_points          = 1200,
    total_assists         = 250,
    total_steals          = 80,
    total_blocks          = 40,
    total_turnovers       = 180,
    total_fouls           = 200,
    home_game_pct         = 0.5
) {

  input_row <- tibble(
    player_name           = "Hypothetical Player",
    season                = 2025L,
    avg_rolling_minutes_3 = avg_rolling_minutes_3,
    load_spike            = load_spike,
    prior_missed_pct      = prior_missed_pct,
    career_season         = career_season,
    ever_injured          = ever_injured,
    total_rebounds        = total_rebounds,
    total_points          = total_points,
    total_assists         = total_assists,
    total_steals          = total_steals,
    total_blocks          = total_blocks,
    total_turnovers       = total_turnovers,
    total_fouls           = total_fouls,
    home_game_pct         = home_game_pct,
    pct_games_missed      = NA_real_,
    any_games_missed      = NA_integer_
  )

  pred <- predict(model, input_row)$.pred
  pred <- max(0, min(1, pred))

  tibble(
    pct_games_missed_predicted = round(pred, 3),
    games_missed_of_82         = round(pred * 82),
    risk_tier = case_when(
      pred >= 0.30 ~ "High Risk",
      pred >= 0.10 ~ "Moderate Risk",
      TRUE         ~ "Low Risk"
    )
  )
}

# Test scenarios
whatif_predict(AI_MODEL_V2,
               avg_rolling_minutes_3 = 38,
               load_spike = 1.3, career_season = 8,
               prior_missed_pct = 0.10, total_rebounds = 600, total_points = 1800)

whatif_predict(AI_MODEL_V2,
               avg_rolling_minutes_3 = 20,
               load_spike = 0.85, career_season = 4,
               prior_missed_pct = 0, total_rebounds = 280, total_points = 900)

whatif_predict(AI_MODEL_V2,
               avg_rolling_minutes_3 = 32,
               load_spike = 1.1, career_season = 12,
               prior_missed_pct = 0.25, ever_injured = 1,
               total_rebounds = 450, total_points = 1400)

###==========================================================================###
###  SECTION 12: PLAYER RISK TABLE — 2024 COHORT                              ###
###==========================================================================###

known_injured_2425 <-
  c("Tyrese Haliburton", "Jayson Tatum", "Damian Lillard",
    "Dejounte Murray",   "James Wiseman", "Isaiah Jackson",
    "Dru Smith",         "DaRon Holmes II")

future_candidates <-
  nba_MODEL_READY_CLEAN %>%
  filter(season == 2024, !player_name %in% known_injured_2425) %>%
  arrange(desc(avg_rolling_minutes_3))

risk_table <-
  AI_MODEL_V2 %>%
  predict(future_candidates) %>%
  bind_cols(future_candidates %>%
              select(player_name, season, avg_rolling_minutes_3,
                     prior_missed_pct, career_season, load_spike)) %>%
  mutate(
    pct_games_missed_predicted = round(pmax(0, pmin(1, .pred)), 3),
    games_missed_of_82         = round(pct_games_missed_predicted * 82),
    risk_tier = case_when(
      pct_games_missed_predicted >= 0.30 ~ "High Risk",
      pct_games_missed_predicted >= 0.10 ~ "Moderate Risk",
      TRUE                               ~ "Low Risk"
    )
  ) %>%
  select(player_name, pct_games_missed_predicted, games_missed_of_82,
         risk_tier, avg_rolling_minutes_3, prior_missed_pct, career_season) %>%
  arrange(desc(pct_games_missed_predicted))

risk_table %>% count(risk_tier)

# Filter to meaningful contributors (>=20 rolling minutes)
risk_table_FILTERED <-
  risk_table %>%
  filter(avg_rolling_minutes_3 >= 20)

risk_table_FILTERED %>% count(risk_tier)

## Top 20 risk visualisation ------------------------------------------------

risk_table_VIZ <-
  risk_table_FILTERED %>%
  slice_head(n = 20) %>%
  mutate(player_name = reorder(player_name, pct_games_missed_predicted)) %>%
  ggplot(aes(x = pct_games_missed_predicted,
             y = player_name,
             fill = risk_tier)) +
  geom_col() +
  geom_vline(xintercept = 0.10, linetype = "dashed",
             color = "orange", linewidth = 0.8) +
  geom_vline(xintercept = 0.30, linetype = "dashed",
             color = "#CE1141", linewidth = 0.8) +
  scale_fill_manual(values = c("High Risk"     = "#CE1141",
                               "Moderate Risk" = "#FF8C00",
                               "Low Risk"      = "#17408B")) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  labs(
    title    = "Top 20 At-Risk NBA Players - Predicted Games Missed (2024 Season)",
    subtitle = "Thresholds: 10% moderate (>8 games), 30% high (>24 games)",
    x        = "Predicted % games missed next season",
    y        = NULL,
    fill     = "Risk tier",
    caption  = "Model: XGBoost V3 | Excludes confirmed 2024-25 injured players"
  ) +
  theme_nba()+
  theme(axis.text.y = element_text(size = 10, color = "black"))

risk_table_VIZ

## Top 5 per risk zone ------------------------------------------------------

risk_showcase_VIZ <-
  risk_table_FILTERED %>%
  group_by(risk_tier) %>%
  slice_head(n = 5) %>%
  ungroup() %>%
  mutate(
    player_name = reorder(player_name, pct_games_missed_predicted),
    label       = paste0(round(pct_games_missed_predicted * 100), "% | ~",
                         games_missed_of_82, " games")
  ) %>%
  ggplot(aes(x = pct_games_missed_predicted,
             y = player_name,
             fill = risk_tier)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = label), hjust = -0.08, size = 3) +
  facet_wrap(~ risk_tier, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("High Risk"     = "#CE1141",
                               "Moderate Risk" = "#FF8C00",
                               "Low Risk"      = "#17408B")) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1.15)) +
  labs(
    title    = "Top 5 Players per Risk Zone - 2024 Season Predictions",
    subtitle = "Predicted % of next season's games missed | >=20 rolling minutes threshold",
    x        = "Predicted % games missed",
    y        = NULL
  ) +
  theme_nba() +
  theme(legend.position = "none",
        strip.text      = element_text(face = "bold", size = 11))

risk_showcase_VIZ

###==========================================================================###
###  SECTION 13: LUKA DONCIC DEEP DIVE                                        ###
###==========================================================================###

# Real-world validation:
# Luka Doncic suffered a Grade 2 left hamstring strain in April 2026 (vs OKC),
# missing the rest of the regular season. He finished with 64 games played
# (~22% of season missed) - squarely in the Moderate Risk tier.
# He averaged 35.8 mins/game and had 5 hamstring injuries since 2024.

luka_from_model <-
  risk_table_FILTERED %>%
  filter(str_detect(player_name, "Doncic|Doncic"))

cat("\n=== Luka Doncic - Model Prediction (2024 season data) ===\n")
luka_from_model %>%
  select(player_name, pct_games_missed_predicted, games_missed_of_82,
         risk_tier, avg_rolling_minutes_3, prior_missed_pct, career_season) %>%
  print()

cat("\n=== What actually happened (April 2026) ===\n")
cat("Injury:      Grade 2 left hamstring strain vs OKC Thunder\n")
cat("Games missed: ~18 regular season games (~22% of season)\n")
cat("Risk tier:   Moderate Risk (>25% threshold)\n")

# What-if with Luka's actual 2024 load profile
# whatif_predict(
#   AI_MODEL_V2,
#   avg_rolling_minutes_3 = 35.8,
#   load_spike            = 1.1,
#   career_season         = 7,
#   prior_missed_pct      = 0.39,
#   ever_injured          = 1,
#   total_rebounds        = 410,
#   total_points          = 1410
# )

whatif_predict(
  AI_MODEL_V2,
  avg_rolling_minutes_3 = 35.8,
  load_spike            = 1.1,
  career_season         = 7,
  prior_missed_pct      = 0.39,
  ever_injured          = 1,
  total_rebounds        = 410,
  total_points          = 1410,
  total_assists         = 385
)

###==========================================================================###
###  SECTION 14: COST OF RECOVERY                                             ###
###==========================================================================###

cor_data <- tibble(
  salary_group = c("Group A\n(<$4M)", "Group B\n($4M-$9M)",
                   "Group C\n(>$9M)",  "Overall\nAverage"),
  cor_millions = c(1.585, 3.830, 7.449, 4.0)
)

cor_VIZ <-
  cor_data %>%
  ggplot(aes(x = salary_group, y = cor_millions, fill = salary_group)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = scales::dollar(cor_millions, suffix = "M")),
            vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = c("#17408B", "#CE1141", "#FF6B00", "#2E8B57")) +
  scale_y_continuous(
    labels = scales::dollar_format(prefix = "$", suffix = "M"),
    limits = c(0, 9)
  ) +
  labs(
    title    = "Mean Cost of Recovery (COR) per Achilles Rupture in the NBA",
    subtitle = "Per-player cost only. Kevin Durant (2019): estimated $60-70M total economic impact.",
    x        = "Player salary group",
    y        = "Mean cost of recovery (USD millions)",
    caption  = "Source: Meadows et al. (2024)"
  ) +
  theme_nba()

cor_VIZ

###==========================================================================###
###  SECTION 15: SAVE                                                         ###
###==========================================================================###

save(
  AI_MODEL_V2,
  wf_v2_final,
  nba_MODEL_READY_CLEAN,
  train_v2,
  test_v2,
  risk_table,
  risk_table_FILTERED,
  whatif_predict,
  pct_missed_dist_VIZ,
  feature_imp_VIZ,
  risk_table_VIZ,
  risk_showcase_VIZ,
  cor_data,
  cor_VIZ,
  model_version_comparison,
  results_v2,
  xgb_last_fit_v2,
  file = "Part2_Predictive_Results.RData"
)

message("Part 2 complete - results saved to Part2_Predictive_Results.RData")

# ── Plain versions for submission.qmd (no custom fonts) ──────────────────────
strip_font <- function(p) {
  p + theme_minimal(base_family = "") +
    theme(text = element_text(family = ""))
}

feature_imp_VIZ_plain <- strip_font(feature_imp_VIZ)
risk_table_VIZ_plain  <- strip_font(risk_table_VIZ)

save(
  AI_MODEL_V2,
  wf_v2_final,
  nba_MODEL_READY_CLEAN,
  train_v2,
  test_v2,
  risk_table,
  risk_table_FILTERED,
  whatif_predict,
  pct_missed_dist_VIZ,
  feature_imp_VIZ,
  risk_table_VIZ,
  risk_showcase_VIZ,
  cor_data,
  cor_VIZ,
  model_version_comparison,
  results_v2,
  xgb_last_fit_v2,
  feature_imp_VIZ,
  feature_imp_VIZ_plain,
  risk_table_VIZ,
  risk_table_VIZ_plain,
  file = "Part2_Predictive_Results.RData"
)
message("✓ Part 2 plain versions saved")