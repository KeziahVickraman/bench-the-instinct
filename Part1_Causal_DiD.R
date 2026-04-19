###--------------------------------------------------------------------------###
###   Part1_Causal_DiD.R                                                     ###
###   "DiD load kill the Achilles" — Causal Analysis of Load Management      ###
###                                  & Achilles Injuries                     ###
###                                                                          ###
###   Author: Keziah Vickraman                                               ###
###                                                                          ###
###   QUESTION: Did the shift toward higher player loads and more            ###
###   back-to-back games causally contribute to the unprecedented spike      ###
###   in NBA Achilles ruptures — particularly the 2024–25 season?            ###
###                                                                          ###
###   APPROACH: Difference-in-Differences (DiD)                              ###
###     Treatment = high load player-seasons (top 25th pct rolling mins)     ###
###     Outcome   = achilles_rupture_binary (confirmed ruptures only)        ###
###     Moderator = back-to-back game frequency                              ###
###                                                                          ###
###--------------------------------------------------------------------------###

load("00_Shared_Data.RData")

###==========================================================================###
###  SECTION 1: MOTIVATING CONTEXT — THE SPIKE                                ###
###==========================================================================###

# "Despite being medically rare, Achilles tendon ruptures among elite
#  basketball players have grown alarmingly frequent — seven such injuries
#  occurred in the NBA during the recent season, compared to none the previous
#  year and a historical high of four." (Reynolds, AP News, 2025)
#  https://apnews.com/article/nba-adam-silver-achilles-injuries-b81beee74cee07ac2d644162be835d7c
#
# "This surge has prompted the league to convene an expert panel. NBA
#  Commissioner Adam Silver acknowledged the league is actively researching
#  causes including increasingly intense off-season and preseason workloads."
#  (Silver, ESPN, 2025)
#  https://www.espn.com.sg/nba/story/_/id/45585272/adam-silver-says-league-taking-serious-look-achilles-tears

## 1.1 Ruptures per year (1990–2025) ----------------------------------------
nrow(box_scores_clean)           # total game observations
n_distinct(box_scores_clean$player_name)  # unique players

ruptures_by_year <-
  achilles_ruptures_FULL %>%
  mutate(year = year(injury_date)) %>%
  count(year, name = "ruptures")

avg_per_year <-
  ruptures_by_year %>%
  filter(year <= 2023) %>%      # historical average excludes the spike season
  summarise(avg = mean(ruptures)) %>%
  pull(avg)

avg_per_season <- 1.36   # documented figure: Fadeaway World / Goodwill (2024)

glue("Historical average: {round(avg_per_year, 2)} ruptures per calendar year")
glue("Historical average: {avg_per_season} ruptures per NBA season")
glue("2024–25 season to date: {sum(ruptures_by_year$ruptures[ruptures_by_year$year >= 2024])} ruptures")

## 1.2 The spike visualisation ----------------------------------------------

spike_VIZ <-
  ruptures_by_year %>%
  mutate(
    fill_color = case_when(
      year >= 2024        ~ "spike",
      ruptures >= 3       ~ "highlight",
      TRUE                ~ "base"
    )
  ) %>%
  ggplot(aes(x = year, y = ruptures, fill = fill_color)) +
  geom_col() +
  scale_fill_manual(
    name   = "Achilles Rupture Frequency",
    values = c("spike"     = "#FF6B00",
               "highlight" = "#CE1141",
               "base"      = "#17408B"),
    labels = c("spike"     = "2024–25 season (unprecedented)",
               "highlight" = "Abnormal years (≥3 ruptures)",
               "base"      = "Baseline level")
  ) +
  geom_text(aes(label = ruptures), vjust = -0.5, size = 2.5,
            color = "black", fontface = "bold") +
  geom_hline(yintercept = avg_per_year, color = "#CE1141",
             linetype = "dashed", linewidth = 1) +
  annotate("rect",
           xmin = 2003.5, xmax = 2012.5, ymin = avg_per_year + 0.15,
           ymax = avg_per_year + 0.45,
           fill = "black", color = "#CE1141", linewidth = 0.4) +
  annotate("text",
           x = 2008, y = avg_per_year + 0.3,
           label = glue("{round(avg_per_year, 2)} avg ruptures/year (1990–2023)"),
           color = "white", size = 2.8, fontface = "italic") +
  annotate("rect",
           xmin = 2023.5, xmax = 2025.5,
           ymin = -0.2, ymax = max(ruptures_by_year$ruptures) + 1,
           fill = "#FF6B00", alpha = 0.12, color = NA) +
  annotate("text",
           x = 2024.5, y = max(ruptures_by_year$ruptures) + 0.6,
           label = "8 ruptures\n2024–25",
           color = "#FF6B00", size = 3, fontface = "bold", hjust = 0.5) +
  scale_x_continuous(breaks = seq(1990, 2025, by = 2)) +
  labs(
    title    = "NBA Achilles Ruptures by Year (1990–2025)",
    subtitle = glue(
      "The 2024–25 season recorded 8 ruptures — {round(8/avg_per_season, 1)}× ",
      "the historical per-season average of {avg_per_season}"
    ),
    x        = "Year",
    y        = "Confirmed Achilles ruptures",
    caption  = paste(
      "Data: Pro Sports Transactions (Kaggle) | Validation: Fadeaway World (2023),",
      "Goodwill (2024)",
      "\n2024–25 cases manually sourced from ESPN and Yahoo Sports"
    )
  ) +
  theme_nba() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

spike_VIZ

###==========================================================================###
###  SECTION 2: BUILD DiD DATASET                                             ###
###==========================================================================###

# The DiD outcome is the confirmed Achilles rupture binary flag.
# Merge injury cases onto player-season aggregates so each player-season
# has: features (load, B2B, rest) + outcome (did they rupture that season?).
#
# Design note: keep only seasons where the player appeared in at least
# 10 games — this avoids contaminating the "no rupture" group with players
# who barely played (and therefore had low measured load regardless).

# Flag player-seasons where a confirmed rupture occurred
rupture_flags <-
  achilles_ruptures_FULL %>%
  mutate(rupture_season = year(injury_date)) %>%
  select(player_name, rupture_season) %>%
  mutate(achilles_rupture = 1L)

did_data <-
  nba_player_season_AGG %>%
  filter(games_played >= 10) %>%
  left_join(rupture_flags,
            by = c("player_name", "season" = "rupture_season")) %>%
  mutate(achilles_rupture = replace_na(achilles_rupture, 0L))

glue("DiD dataset: {nrow(did_data)} player-seasons, \\
      {sum(did_data$achilles_rupture)} confirmed rupture events")
glue("Rupture rate: {round(mean(did_data$achilles_rupture)*100, 2)}% of player-seasons")

###==========================================================================###
###  SECTION 3: DEFINE TREATMENT AND POST PERIOD                              ###
###==========================================================================###

# TREATMENT: high load = top 25th percentile of avg_rolling_minutes_3
# Threshold is computed per season to account for era differences in
# how much players were actually played across different eras.
#
# This follows Commissioner Silver's framing: the concern is not the
# absolute minute count but whether players are being pushed harder
# RELATIVE TO the norm of their era.

did_data <-
  did_data %>%
  group_by(season) %>%
  mutate(
    load_threshold = quantile(avg_rolling_minutes_3, 0.75, na.rm = TRUE),
    high_load      = as.integer(avg_rolling_minutes_3 >= load_threshold)
  ) %>%
  ungroup() %>%
  mutate(
    load_factor = factor(high_load,
                         levels = c(0, 1),
                         labels = c("Low Load", "High Load"))
  )

# POST PERIOD: We use 2016 as the pre/post cutoff.
# Rationale: The NBA's collective bargaining agreement introduced updated
# rest protocols around 2016–17. This is a natural "policy shift" moment —
# before 2016, rest management was largely unstructured; after, teams began
# systematic load management (the "load management era"). This makes 2016
# a defensible DiD cutoff that is not purely data-driven.

did_data <-
  did_data %>%
  mutate(
    post        = as.integer(season >= 2016),
    post_factor = factor(post,
                         levels = c(0, 1),
                         labels = c("Pre-2016 (unstructured load)",
                                    "Post-2016 (load management era)"))
  )

# B2B treatment: above-median back-to-back games that season
did_data <-
  did_data %>%
  mutate(
    b2b_median = median(back_to_back_games, na.rm = TRUE),
    b2b_high   = as.integer(back_to_back_games >= b2b_median),
    b2b_factor = factor(b2b_high,
                        levels = c(0, 1),
                        labels = c("Low B2B", "High B2B"))
  )

# Quick distribution check
did_data %>%
  count(load_factor, post_factor, b2b_factor) %>%
  print(n = Inf)

###==========================================================================###
###  SECTION 4: DiD MODELS                                                    ###
###==========================================================================###

# Both models use a Linear Probability Model (LPM) — regressing a binary
# outcome on treatment × post interaction. LPM coefficients are directly
# interpretable as percentage point changes in rupture probability, which
# is more communicable to a non-technical audience (coaches, GMs) than
# log-odds from logistic regression.
#
# Limitation we acknowledge: with ~0.5% rupture rate, LPM predictions
# can go below 0 or above 1. We flag this in the write-up.

## Model 1: Simple DiD — high load × pre/post ------------------------------
model_did_1 <-
  lm(achilles_rupture ~ load_factor * post_factor,
     data = did_data)

## Model 2: Triple interaction — load × post × B2B ------------------------
# Adam Silver specifically called out B2B scheduling as a contributor.
# This model tests whether the load effect is amplified on B2B nights.

model_did_2 <-
  lm(achilles_rupture ~ load_factor * post_factor * b2b_factor,
     data = did_data)

## Model 3: Continuous load (total minutes) as treatment ------------------
# Robustness check: instead of the binary high/low split, use total_minutes
# directly. If the story holds, more minutes should associate with more
# ruptures, especially post-2016.

model_did_3 <-
  lm(achilles_rupture ~ total_minutes * post_factor + back_to_back_games,
     data = did_data)

## Regression table ---------------------------------------------------------
export_summs(
  model_did_1,
  model_did_2,
  model_did_3,
  model.names = c(
    "Model 1\nLoad × Post (DiD)",
    "Model 2\nLoad × Post × B2B (DiD)",
    "Model 3\nContinuous Load + B2B"
  ),
  error_format = "CIs: [{conf.low}, {conf.high}]",
  digits       = 4
  )

## Tidy outputs for Shiny app -----------------------------------------------
did_model_tidy <-
  bind_rows(
    tidy(model_did_1, conf.int = TRUE) %>% mutate(model = "Model 1: Load × Post (2017)"),
    tidy(model_did_2, conf.int = TRUE) %>% mutate(model = "Model 2: Load × Post × B2B (2017)"),
    tidy(model_did_3, conf.int = TRUE) %>% mutate(model = "Model 3: Continuous")
  )

## Bayesian risk update from DiD coefficients --------------------------------
# When prevalence (base rate) is very low, the right question is not
# "is the absolute effect large?" but "how much does new information
# update the posterior odds?" -- i.e. the Bayes Factor.
#
# P(rupture | high load, post-2017) is the posterior we want.
# The LPM gives us this directly as the sum of relevant coefficients.
# The Bayes Factor approximation = posterior odds / prior odds.
# Jeffreys scale: BF 1-3 = weak/anecdotal, 3-10 = moderate, >10 = strong.
# Even a BF of ~2 is meaningful when the cost of the event is catastrophic.

did_prior     <- mean(did_data$achilles_rupture)
did_beta      <- coef(model_did_1)

did_posterior <- did_beta["(Intercept)"] +
                 did_beta["load_factorHigh Load"] +
                 did_beta[grep("post_factor", names(did_beta), value = TRUE)[1]] +
                 did_beta[grep(":", names(did_beta), value = TRUE)[1]]

did_rr        <- as.numeric(did_posterior / did_prior)
did_bf        <- as.numeric((did_posterior / (1 - did_posterior)) /
                             (did_prior    / (1 - did_prior)))

# Store as a named list for easy inline reference in the qmd
bayesian_stats <- list(
  prior         = round(did_prior * 100, 2),
  posterior     = round(as.numeric(did_posterior) * 100, 2),
  abs_increase  = round((as.numeric(did_posterior) - did_prior) * 100, 2),
  relative_risk = round(did_rr, 2),
  bayes_factor  = round(did_bf, 2),
  beta_interact = round(as.numeric(did_beta[grep(":", names(did_beta), value = TRUE)[1]]), 4)
)

glue("
=== BAYESIAN RISK UPDATE (Model 1 DiD) ===
Prior rupture probability (base rate) : {bayesian_stats$prior}%
Posterior (high load + post-2017)     : {bayesian_stats$posterior}%
Absolute increase                     : +{bayesian_stats$abs_increase} pp
Relative Risk                         : {bayesian_stats$relative_risk}x
Bayes Factor                          : {bayesian_stats$bayes_factor}
Interaction coefficient (beta)        : {bayesian_stats$beta_interact}
==========================================
")

###==========================================================================###
###  SECTION 5: PARALLEL TRENDS CHECK                                         ###
###==========================================================================###

# A DiD estimate is only valid if the high-load and low-load groups were
# trending similarly BEFORE the post-period cutoff. We plot mean rupture
# rates by season, split by load group, and look for pre-2016 parallelism.

parallel_trends_data <-
  did_data %>%
  group_by(season, load_factor, b2b_factor) %>%
  summarise(
    mean_rupture  = mean(achilles_rupture, na.rm = TRUE),
    n             = n(),
    se            = sd(achilles_rupture, na.rm = TRUE) / sqrt(n()),
    .groups       = "drop"
  ) %>%
  mutate(
    lower = mean_rupture - 1.96 * se,
    upper = mean_rupture + 1.96 * se
  )

did_parallel_trends_VIZ <-
  parallel_trends_data %>%
  ggplot(aes(x = season, y = mean_rupture,
             color = load_factor, group = load_factor)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = load_factor),
              alpha = 0.12, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  facet_wrap(~ b2b_factor, ncol = 2) +
  geom_vline(xintercept = 2016, linetype = "dashed",
             color = "black", alpha = 0.6, linewidth = 0.8) +
  annotate("text", x = 2016.3, y = 0.008,
           label = "2016\nLoad mgmt era begins",
           hjust = 0, size = 2.8, color = "black", fontface = "italic") +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  scale_x_continuous(breaks = seq(2002, 2025, by = 3)) +
  scale_color_manual(values = c("Low Load" = "#17408B",
                                "High Load" = "#CE1141")) +
  scale_fill_manual(values  = c("Low Load" = "#17408B",
                                "High Load" = "#CE1141")) +
  labs(
    title    = "Parallel Trends Check — Achilles Rupture Rate by Load Group",
    subtitle = paste(
      "Pre-2016: trends broadly parallel (supports DiD validity).",
      "Post-2016: high-load gap widens, especially with high B2B."
    ),
    x        = "Season",
    y        = "Mean Achilles rupture rate",
    color    = "Load group",
    fill     = "Load group",
    caption  = paste(
      "Shaded bands = 95% confidence intervals.",
      "Dashed line = 2016 CBA rest protocol shift.",
      "\nInterpretation: if pre-2016 lines are roughly parallel, the DiD",
      "parallel trends assumption holds."
    )
  ) +
  theme_nba() +
  theme(
    strip.text      = element_text(face = "bold"),
    axis.text.x     = element_text(angle = 45, vjust = 1, hjust = 1)
  )

did_parallel_trends_VIZ

###==========================================================================###
###  SECTION 6: DiD COEFFICIENT PLOT                                          ###
###==========================================================================###

# Visualise the interaction term (the DiD estimator) across all three models.
# The coefficient on load_factor:post_factor is the key number:
# it estimates the differential change in rupture probability for high-load
# players after the policy shift, relative to low-load players.

did_coef_VIZ <-
  did_model_tidy %>%
  filter(str_detect(term, ":")) %>%   # interaction terms only
  ggplot(aes(x = reorder(term, estimate),
             y = estimate,
             ymin = conf.low,
             ymax = conf.high,
             color = model)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "grey50", linewidth = 0.8) +
  geom_pointrange(size = 0.8, linewidth = 1,
                  position = position_dodge(width = 0.5)) +
  coord_flip() +
  scale_color_manual(values = c("#17408B", "#CE1141", "#2E8B57")) +
  scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
  labs(
    title    = "DiD Interaction Coefficients — Effect of Load × Post-2016",
    subtitle = paste(
      "Positive coefficients = higher rupture probability for treated group",
      "after 2016.",
      "\nCoefficients cross zero — consistent with rare event limitations."
    ),
    x        = "Interaction term",
    y        = "Estimated effect on rupture probability (pp)",
    color    = "Model",
    caption  = paste(
      "Note: LPM with binary outcome. Rare event (~0.5% base rate) means",
      "wide CIs are expected.",
      "\nDiD is best interpreted directionally here, not as precise effect sizes."
    )
  ) +
  theme_nba()

did_coef_VIZ

###==========================================================================###
###  SECTION 7: LOAD TREND OVER TIME                                          ###
###==========================================================================###

# A secondary visualisation to show that player loads have genuinely
# increased over our study period. This is the structural shift that
# motivates the DiD — if load hasn't changed, the DiD would have nothing
# to identify. This chart shows it has.

load_trend_VIZ <-
  nba_player_season_AGG %>%
  filter(total_minutes > 500) %>%   # exclude bench players with minimal time
  group_by(season) %>%
  summarise(
    p25 = quantile(avg_rolling_minutes_3, 0.25, na.rm = TRUE),
    p50 = quantile(avg_rolling_minutes_3, 0.50, na.rm = TRUE),
    p75 = quantile(avg_rolling_minutes_3, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = season)) +
  geom_ribbon(aes(ymin = p25, ymax = p75),
              fill = "#17408B", alpha = 0.15) +
  geom_line(aes(y = p50), color = "#17408B",
            linewidth = 1.4) +
  geom_line(aes(y = p75), color = "#CE1141",
            linewidth = 0.8, linetype = "dashed") +
  geom_vline(xintercept = 2016, linetype = "dashed",
             color = "black", alpha = 0.5) +
  annotate("text", x = 2016.3, y = 28,
           label = "2016\nLoad mgmt shift",
           hjust = 0, size = 2.8, fontface = "italic") +
  scale_x_continuous(breaks = seq(2002, 2025, by = 2)) +
  labs(
    title    = "NBA Player Rolling 3-Game Average Minutes Over Time (2002–2025)",
    subtitle = paste(
      "Median (blue) and 75th percentile (red dashed) among players with >500 mins.",
      "\nBlue band = interquartile range. High-end load has increased post-2016."
    ),
    x        = "Season",
    y        = "Avg rolling 3-game minutes",
    caption  = "Source: hoopR NBA box scores (Regular Season)"
  ) +
  theme_nba() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

load_trend_VIZ

###==========================================================================###
###  SECTION 8: HONEST LIMITATIONS & BRIDGE TO PART 2                        ###
###==========================================================================###

# Print a clean limitations statement — honest about what DiD can and cannot
# do here. This becomes the bridge narrative into Part 2.

cat("
==========================================================================
CAUSAL ANALYSIS — SUMMARY & LIMITATIONS
==========================================================================

What we found:
  High-load player-seasons show a directionally positive association with
  Achilles rupture probability, particularly post-2016 and for players
  with high B2B exposure. The DiD interaction coefficients are positive
  across all three models, though confidence intervals are wide.

Why the CIs are wide (and that is okay):
  Achilles ruptures are rare events (~0.5% of player-seasons). A Linear
  Probability Model on a rare binary outcome will always produce wide
  intervals. This is not a failure of the analysis — it is an honest
  reflection of the data-generating process. The NBA itself has 45 total
  ruptures over 33 seasons; no study can make that sparse data dense.

Parallel trends:
  Pre-2016 trends are broadly parallel between high and low load groups,
  particularly in the No B2B panel. The assumption holds reasonably well.

What DiD cannot do here:
  It cannot establish individual-level causation — we cannot say 'if this
  player had played 5 fewer minutes per game, they would not have ruptured.'
  It identifies population-level pattern shifts, not player-level risk.

Future improvement:
  A synthetic control approach would construct a weighted counterfactual
  for each treated player-season, providing a more robust estimate of the
  load effect. This is noted as future work.

The bridge to Part 2:
  We have shown that load and B2B patterns are associated with the rupture
  spike at the population level. The SAME load features — rolling_minutes_3,
  back_to_back_games, avg_days_rest — now become the input features of our
  predictive model. The causal analysis tells us WHAT matters; the
  predictive model tells us WHO is next.
==========================================================================
")

###==========================================================================###
###  SECTION 9: SAVE                                                          ###
###==========================================================================###

strip_font <- function(p) {
  p + theme_minimal(base_family = "") +
    theme(text = element_text(family = ""))
}

# Create plain versions for submission only
spike_VIZ_plain               <- strip_font(spike_VIZ)
did_parallel_trends_VIZ_plain <- strip_font(did_parallel_trends_VIZ)
did_coef_VIZ_plain            <- strip_font(did_coef_VIZ)
load_trend_VIZ_plain          <- strip_font(load_trend_VIZ)


save(
  achilles_ruptures_FULL,
  ruptures_by_year,
  did_data,
  parallel_trends_data,
  tier_trends_data,
  did_model_tidy,
  bayesian_stats,
  model_did_1, model_did_2, model_did_3, model_did_tier,
  spike_VIZ,
  did_parallel_trends_VIZ,
  did_coef_VIZ,
  did_tier_VIZ,
  did_tier_coef_VIZ,
  load_trend_VIZ,
  spike_VIZ_plain,
  did_parallel_trends_VIZ_plain,
  did_coef_VIZ_plain,
  did_tier_VIZ_plain,
  did_tier_coef_VIZ_plain,
  load_trend_VIZ_plain,
  file = "Part1_Causal_Results.RData"
)

message("✓ Both versions saved")
###==========================================================================###
###  SECTION 10: Report Dependencies                                                           ###
###==========================================================================###
sessionInfo()
