###--------------------------------------------------------------------------###
###   Capstone_Shiny_PREDICTION.R                                             ###
###   "Bench the Instinct" — NBA Player Risk Dashboard                       ###
###                                                                           ###
###   Audience: Coaches, GMs, medical staff                                  ###
###   Purpose:  Data-driven load management — predict who is at risk         ###
###             before the injury happens                                     ###
###                                                                           ###
###   Tabs:                                                                   ###
###     1. The Spike       — why this tool exists (motivating context)       ###
###     2. Who's Next?     — player risk table, filterable                   ###
###     3. Bench the Instinct — what-if load simulator                       ###
###     4. Model Info      — honest diagnostics for the technically curious  ###
###--------------------------------------------------------------------------###

# install.packages("rsconnect")

# pacman::p_load(shiny, tidyverse, scales, ggthemes, DT, glue, vip, rsconnect)
library(shiny)
library(tidyverse)
library(scales)
library(ggthemes)
library(DT)
library(glue)
library(vip)
library(showtext)

library(xgboost)
library(tidymodels)
library(workflows)

load("Shiny_App_Data.RData")
AI_MODEL_V2 <- readRDS("AI_MODEL_V2.rds")
model_did_1 <- readRDS("model_did_1.rds")
model_did_2 <- readRDS("model_did_2.rds")
model_did_3 <- readRDS("model_did_3.rds")

# # Save a slim version with ONLY what the Shiny app uses
# save(
#   # Part 1 — only the plots and models the app displays
#   spike_VIZ,
#   load_trend_VIZ,
#   did_parallel_trends_VIZ,
#   did_coef_VIZ,
#   achilles_ruptures_FULL,
#   model_did_1, model_did_2, model_did_3,
#   
#   # Part 2 — only what the app displays
#   AI_MODEL_V2,
#   risk_table_FILTERED,
#   whatif_predict,
#   feature_imp_VIZ,
#   pct_missed_dist_VIZ,
#   cor_VIZ,
#   model_version_comparison,
#   
#   file = "Shiny_App_Data.RData"
# )
# 
# # Check the size
# file.size("Shiny_App_Data.RData") / 1e6  # should be much smaller
# 
# # Check size of each object
# sizes <- sapply(c("AI_MODEL_V2", "spike_VIZ", "load_trend_VIZ", 
#                   "did_parallel_trends_VIZ", "did_coef_VIZ",
#                   "achilles_ruptures_FULL", "model_did_1", 
#                   "model_did_2", "model_did_3",
#                   "risk_table_FILTERED", "whatif_predict",
#                   "feature_imp_VIZ", "pct_missed_dist_VIZ", 
#                   "cor_VIZ", "model_version_comparison"),
#                 function(x) object.size(get(x)))
# 
# sort(sizes, decreasing = TRUE) %>% 
#   as.data.frame() %>% 
#   tibble::rownames_to_column("object") %>%
#   mutate(size_mb = round(`.` / 1e6, 2)) %>%
#   select(object, size_mb)
# 
# # Save model with xz compression (much smaller)
# saveRDS(AI_MODEL_V2,  "AI_MODEL_V2.rds",  compress = "xz")
# saveRDS(model_did_1,  "model_did_1.rds",  compress = "xz")
# saveRDS(model_did_2,  "model_did_2.rds",  compress = "xz")
# saveRDS(model_did_3,  "model_did_3.rds",  compress = "xz")
# 
# # Check sizes
# cat(file.size("AI_MODEL_V2.rds")  / 1e6, "MB - AI_MODEL_V2\n")
# cat(file.size("model_did_1.rds")  / 1e6, "MB - model_did_1\n")
# cat(file.size("model_did_2.rds")  / 1e6, "MB - model_did_2\n")
# cat(file.size("model_did_3.rds")  / 1e6, "MB - model_did_3\n")
# 
# # Save everything else as slim RData
# save(
#   spike_VIZ, load_trend_VIZ,
#   did_parallel_trends_VIZ, did_coef_VIZ,
#   achilles_ruptures_FULL,
#   risk_table_FILTERED, whatif_predict,
#   feature_imp_VIZ, pct_missed_dist_VIZ,
#   cor_VIZ, model_version_comparison,
#   file = "Shiny_App_Data.RData"
# )
# 
# cat(file.size("Shiny_App_Data.RData") / 1e6, "MB - Shiny_App_Data\n")


# Manually define theme_nba() here since we can't source 00_Data_Setup.R
library(showtext)
font_add_google("Roboto Condensed", "roboto")
showtext_auto()

theme_nba <- function() {
  theme_minimal(base_family = "roboto") +
    theme(
      plot.background  = element_rect(fill = "#FFFFFF", color = NA),
      panel.grid.major = element_line(color = "#E0E0E0"),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = 16, color = "black"),
      plot.subtitle    = element_text(size = 12, color = "gray30"),
      plot.caption     = element_text(size = 8, color = "black", face = "italic", hjust = 0),
      axis.title       = element_text(face = "bold", size = 10),
      axis.text        = element_text(size = 9),
      legend.position  = "top",
      legend.direction = "horizontal",
      legend.title     = element_text(face = "bold", size = 10, color = "black"),
      legend.text      = element_text(size = 9, color = "black")
    )
}

# # For App Deployment 
# rsconnect::setAccountInfo(name='7tmnpt-keziah-vickraman', 
#                           token='E9EC87AA0B6B04ABB86764723BE773D6', 
#                           secret='BFcVFEZGpCzt6KZPv/7iCkPpuyDKkru958+8tyPX')
# rsconnect::deployApp(
#   appName       = "nba-achilles",
#   appFiles      = c(
#     "Capstone_Shiny_PREDICTION.R",
#     "Shiny_App_Data.RData",
#     "AI_MODEL_V2.rds",
#     "model_did_1.rds",
#     "model_did_2.rds",
#     "model_did_3.rds"
#   ),
#   appPrimaryDoc = "Capstone_Shiny_PREDICTION.R"
# )
# ── Deployment complete ──────────────────────────────────────────────────
# ✔ Successfully deployed to <https://7tmnpt-keziah-vickraman.shinyapps.io/nba-achilles/>

###==========================================================================###
###  UI                                                                        ###
###==========================================================================###

ui <- fluidPage(

  tags$head(
    tags$link(
      href = "https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&family=Roboto:wght@300;400;500&display=swap",
      rel  = "stylesheet"
    ),
    tags$style(HTML("
      * { box-sizing: border-box; }
      body {
        font-family: 'Roboto', sans-serif;
        background: #F2F4F8;
        color: #1A1A2E;
      }
      .navbar-default {
        background-color: #17408B !important;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      }
      .navbar-default .navbar-brand,
      .navbar-default .navbar-nav > li > a {
        color: #FFFFFF !important;
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: 700;
        letter-spacing: 0.03em;
      }
      .navbar-default .navbar-nav > .active > a,
      .navbar-default .navbar-nav > li > a:hover {
        background-color: #CE1141 !important;
        color: #FFFFFF !important;
      }
      .hero {
        background: linear-gradient(135deg, #17408B 0%, #0d2b5e 100%);
        color: white;
        padding: 40px 30px 30px;
        margin-bottom: 30px;
        border-radius: 0 0 12px 12px;
      }
      .hero h1 {
        font-family: 'Roboto Condensed', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        color: white;
      }
      .hero p { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin: 0; }
      .hero .tagline { color: #CE1141; font-weight: 700; font-size: 1rem; margin-bottom: 8px; }
      .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
        border-top: 4px solid #17408B;
      }
      .stat-card.red   { border-top-color: #CE1141; }
      .stat-card.orange{ border-top-color: #FF8C00; }
      .stat-card h2    { font-size: 2.4rem; font-weight: 700; margin: 0; color: #17408B; }
      .stat-card.red h2   { color: #CE1141; }
      .stat-card.orange h2{ color: #FF8C00; }
      .stat-card p     { margin: 4px 0 0; color: #666; font-size: 0.9rem; }
      .result-card {
        background: white;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
      }
      .result-card h3  { margin-top: 0; font-family: 'Roboto Condensed', sans-serif; }
      .metric-big      { font-size: 2.8rem; font-weight: 700; margin: 4px 0; }
      .metric-label    { font-size: 0.85rem; color: #888; text-transform: uppercase;
                         letter-spacing: 0.05em; }
      .well {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
      }
      .callout-blue {
        background: #E8F1FB;
        border-left: 4px solid #17408B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.9rem;
      }
      .callout-red {
        background: #FEF0F3;
        border-left: 4px solid #CE1141;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.9rem;
      }
      .tab-content     { padding-top: 20px; }
      .btn-danger      { background-color: #CE1141 !important;
                         border-color: #CE1141 !important; }
      .btn-danger:hover{ background-color: #a50e33 !important; }
      hr { border-color: #E0E0E0; }
      h3 { font-family: 'Roboto Condensed', sans-serif; color: #17408B; }
      h4 { color: #CE1141; }
    "))
  ),

  # Hero banner
  div(class = "hero",
    fluidRow(
      column(8,
        p(class = "tagline", "🏀 NBA ACHILLES LOAD MANAGEMENT STUDY — KEZIAH VICKRAMAN, 2025"),
        h1("Bench the Instinct"),
        p(paste(
          "We know load caused the spike. Now we know who's next.",
          "The question is whether the league acts on data —",
          "or waits for the next Tatum."
        ))
      ),
      column(4,
        div(style = "text-align: right; padding-top: 10px;",
          div(class = "stat-card red", style = "display:inline-block; width:140px; margin-right:10px;",
            h2("8"),
            p("ruptures 2024–25")
          ),
          div(class = "stat-card", style = "display:inline-block; width:140px;",
            h2("1.36"),
            p("historical avg/season")
          )
        )
      )
    )
  ),

  navbarPage(
    title = "",
    windowTitle = "Bench the Instinct — NBA Risk Dashboard",

    ###----------------------------------------------------------------------###
    ###  TAB 1: THE SPIKE                                                     ###
    ###----------------------------------------------------------------------###
    tabPanel("🦵 Did Load Kill the Achilles?",

      fluidRow(
        column(12,
          h3("The 2024–25 Season Changed Everything"),
          div(class = "callout-red",
            strong("8 Achilles ruptures in one season."),
            " Nearly 6× the historical average of 1.36 per season.",
            " Tatum. Haliburton. Lillard. Damian Murray. The names keep coming.",
            " NBA Commissioner Adam Silver called it unprecedented and convened an expert panel.",
            " This tool is our answer to his question: is this just bad luck?"
          )
        )
      ),

      fluidRow(column(12, plotOutput("spike_plot", height = "420px"))),

      fluidRow(column(12, br(), plotOutput("load_trend_plot", height = "360px"))),

      fluidRow(
        column(12,
          br(),
          div(class = "callout-blue",
            strong("What the load trend tells us:"),
            " Median player minutes have risen post-2016 while elite player load plateaued.",
            " The gap is widening — high-usage players just outside formal load management",
            " protocols are the most structurally exposed group.",
            " Load management protected the stars. It left everyone else behind."
          ),
          br(),
          h4("Confirmed Achilles Rupture Cases (1990–2025)"),
          DT::dataTableOutput("rupture_table")
        )
      )
    ),

    ###----------------------------------------------------------------------###
    ###  TAB 2: WHO'S NEXT?                                                  ###
    ###----------------------------------------------------------------------###
    tabPanel("🔮 Who's Next?",

      fluidRow(
        column(12,
          h3("Player Risk Rankings — 2024 Season"),
          div(class = "callout-blue",
            strong("How to read this:"),
            " Each bar shows the predicted % of next season's games a player will miss,",
            " based on their 2024 load profile and injury history.",
            " Restricted to players averaging ≥20 rolling minutes — meaningful contributors only.",
            br(),
            strong("High Risk (>30%):"), " expected to miss 24+ games.",
            strong(" Moderate Risk (10–30%):"), " expected to miss 8–24 games.",
            strong(" Low Risk (<10%):"), " likely to be available."
          )
        )
      ),

      fluidRow(
        column(3,
          wellPanel(
            h4("Filters", style = "color: #17408B;"),

            sliderInput("risk_threshold",
                        "Min predicted % games missed:",
                        min = 0, max = 1, value = 0.10, step = 0.05),

            selectInput("risk_tier_filter",
                        "Risk tier:",
                        choices  = c("All", "High Risk",
                                     "Moderate Risk", "Low Risk"),
                        selected = "All"),

            sliderInput("min_rolling_mins",
                        "Min avg rolling minutes:",
                        min = 0, max = 40, value = 20, step = 1),

            hr(),
            div(class = "callout-blue",
              strong("Top driver:"),
              " Avg rolling 3-game minutes (lagged).",
              " More recent load = higher predicted absence."
            )
          )
        ),

        column(9,
          plotOutput("risk_table_plot", height = "540px")
        )
      ),

      fluidRow(
        column(12,
          br(),
          h4("Full Sortable Risk Table"),
          DT::dataTableOutput("risk_datatable")
        )
      )
    ),

    ###----------------------------------------------------------------------###
    ###  TAB 3: BENCH THE INSTINCT                                           ###
    ###----------------------------------------------------------------------###
    tabPanel("⚡ Bench the Instinct",

      fluidRow(
        column(12,
          h3("Load Simulator — Model the Risk Before It Happens"),
          div(class = "callout-red",
            strong("This is the tool coaches have been missing."),
            " Instead of relying on gut feel to decide when to rest a player,",
            " enter their load profile and get a predicted % of games missed.",
            " Compare scenarios: what happens if you give them 3 extra rest days?",
            " What if you reduce their rolling minutes from 36 to 28?",
            " The data answers. You decide."
          )
        )
      ),

      fluidRow(

        column(4,
          wellPanel(
            h4("Player Load Profile", style = "color: #17408B;"),

            sliderInput("wi_rolling_mins",
                        "Avg rolling 3-game minutes (lagged):",
                        min = 0, max = 42, value = 28, step = 0.5),
            p(em("Higher = more recent workload intensity"),
              style = "font-size:0.8rem; color:#888; margin-top:-8px;"),

            sliderInput("wi_load_spike",
                        "Load spike (vs own season average):",
                        min = 0, max = 2, value = 1.0, step = 0.05),
            p(em("1.0 = playing at their norm | >1.0 = above norm"),
              style = "font-size:0.8rem; color:#888; margin-top:-8px;"),

            sliderInput("wi_prior_missed",
                        "Prior season % games missed:",
                        min = 0, max = 1, value = 0, step = 0.05),
            p(em("0 = played every game last season"),
              style = "font-size:0.8rem; color:#888; margin-top:-8px;"),

            sliderInput("wi_career",
                        "Career season (years in NBA):",
                        min = 1, max = 22, value = 5, step = 1),
            p(em("Risk peaks at seasons 4–11 (the heavy usage years)"),
              style = "font-size:0.8rem; color:#888; margin-top:-8px;"),

            checkboxInput("wi_ever_injured",
                          "Has prior injury history?",
                          value = FALSE),

            hr(),
            h5("Performance profile (optional)",
               style = "color: #666;"),

            sliderInput("wi_rebounds", "Total rebounds:",
                        min = 0, max = 1000, value = 400, step = 25),
            sliderInput("wi_points",   "Total points:",
                        min = 0, max = 3000, value = 1200, step = 50),

            actionButton("run_whatif", "🔍 Run Prediction",
                         class = "btn btn-danger btn-block",
                         style = "font-size: 1.1rem; padding: 12px; margin-top: 10px;")
          )
        ),

        column(8,
          br(),
          uiOutput("whatif_result_card"),
          br(),
          plotOutput("whatif_comparison_plot", height = "340px"),
          br(),
          div(class = "callout-blue",
            strong("How to use this for load management decisions:"),
            br(),
            "1. Enter a player's current rolling minutes and prior missed games.",
            br(),
            "2. Note the predicted % missed.",
            br(),
            "3. Reduce rolling minutes by 4–6 and re-run.",
            br(),
            "4. If the predicted risk drops significantly, you have a data-backed",
            " case for resting them. That is the conversation to have with the player."
          )
        )
      )
    ),

    ###----------------------------------------------------------------------###
    ###  TAB 4: MODEL INFO                                                   ###
    ###----------------------------------------------------------------------###
    tabPanel("📋 Model Info",

      fluidRow(
        column(12,
          h3("About the Model"),
          div(class = "callout-blue",
            strong("For the technically curious."),
            " This section documents what the model is, how it was built,",
            " and — importantly — what it cannot do.",
            " Honest diagnostics matter as much as strong predictions."
          )
        )
      ),

      fluidRow(
        column(6,
          div(class = "stat-card",
            h2("0.747"), p("R² on held-out test set (2022–2025)")
          )
        ),
        column(3,
          div(class = "stat-card red",
            h2("0.133"), p("RMSE (~11 games on 82-game scale)")
          )
        ),
        column(3,
          div(class = "stat-card orange",
            h2("0.100"), p("MAE (~8 games on 82-game scale)")
          )
        )
      ),

      fluidRow(
        column(12,
          h4("Model Version History — Why R² Went Down (And That's Good)"),
          p(em(paste(
            "Each version corrected a source of data leakage or bias.",
            "Lower R² reflects a more honest model, not a worse one."
          ))),
          tableOutput("model_versions_table"),
          br()
        )
      ),

      fluidRow(
        column(6,
          h4("Feature Importance"),
          p("What the model learned drives risk — top 4 are all load features."),
          plotOutput("feature_imp_plot", height = "360px")
        ),
        column(6,
          h4("Outcome Distribution"),
          p("Most players miss 15–30% of games. The model predicts across this full range."),
          plotOutput("dist_plot", height = "360px")
        )
      ),

      fluidRow(
        column(12,
          br(),
          h4("What This Model Cannot Do"),
          div(class = "callout-red",
            tags$ul(
              tags$li(strong("It cannot predict specific injury type."),
                " pct_games_missed captures all absence reasons — Achilles, other injuries, rest."),
              tags$li(strong("It does not have biomechanical data."),
                " Tendon stiffness, landing mechanics, and fatigue indicators are not in box scores."),
              tags$li(strong("It is not real-time."),
                " Predictions are based on season-level aggregates, not rolling game-by-game updates."),
              tags$li(strong("It is a tool, not an oracle."),
                " Use it to start the conversation with players and medical staff — not to end it.")
            )
          ),
          br(),
          h4("The Economic Case"),
          p(paste(
            "Each prevented Achilles rupture saves millions.",
            "For a max-contract player, the total economic impact exceeds $60–70M.",
            "A data-driven load management programme costs a fraction of one injury."
          )),
          plotOutput("cor_plot", height = "300px")
        )
      )
    )
  )
)

###==========================================================================###
###  SERVER                                                                    ###
###==========================================================================###

server <- function(input, output, session) {

  ## Tab 1 ------------------------------------------------------------------

  output$spike_plot      <- renderPlot({ spike_VIZ })
  output$load_trend_plot <- renderPlot({ load_trend_VIZ })

  output$rupture_table <- DT::renderDataTable({
    achilles_ruptures_FULL %>%
      select(player_name, injury_date, injury_desc, injury_severity) %>%
      arrange(desc(injury_date)) %>%
      DT::datatable(
        options  = list(pageLength = 10, scrollX = TRUE,
                        dom = "ftp"),
        rownames = FALSE,
        colnames = c("Player", "Date", "Description", "Severity")
      )
  })

  ## Tab 2 ------------------------------------------------------------------

  filtered_risk <- reactive({
    risk_table_FILTERED %>%
      filter(
        pct_games_missed_predicted >= input$risk_threshold,
        avg_rolling_minutes_3      >= input$min_rolling_mins,
        if (input$risk_tier_filter == "All") TRUE
        else risk_tier == input$risk_tier_filter
      )
  })

  output$risk_table_plot <- renderPlot({

    df <- filtered_risk() %>%
      slice_head(n = 20) %>%
      mutate(player_name = reorder(player_name,
                                    pct_games_missed_predicted))

    if (nrow(df) == 0) {
      ggplot() +
        annotate("text", x = 0.5, y = 0.5,
                 label = "No players match current filters.\nTry lowering the thresholds.",
                 size = 5, color = "grey50", hjust = 0.5) +
        theme_void()
    } else {
      df %>%
        ggplot(aes(x    = pct_games_missed_predicted,
                   y    = player_name,
                   fill = risk_tier)) +
        geom_col(width = 0.75) +
        geom_text(aes(label = paste0(round(pct_games_missed_predicted * 100), "%")),
                  hjust = -0.15, size = 3.2, fontface = "bold") +
        geom_vline(xintercept = 0.10, linetype = "dashed", 
                   color = "#FF8C00", linewidth = 0.9) +
        geom_vline(xintercept = 0.30, linetype = "dashed",
                   color = "#CE1141", linewidth = 0.9) +
        scale_fill_manual(
          values = c("High Risk"     = "#CE1141",
                     "Moderate Risk" = "#FF8C00",
                     "Low Risk"      = "#17408B")
        ) +
        scale_x_continuous(labels = percent_format(accuracy = 1),
                           limits = c(0, 1.05),
                           expand = c(0, 0)) +
        labs(
          title    = "Predicted % Games Missed — Top At-Risk Players (2024 Season)",
          subtitle = "10% threshold = moderate (>8 games) | 30% threshold = high (>24 games) | Labels show predicted %",
          x        = "Predicted % of next season's games missed",
          y        = NULL,
          fill     = "Risk tier",
          caption  = "Model: XGBoost regression (V3) | Restricted to ≥20 rolling minutes"
        ) +
        theme_nba() +
        theme(legend.position = "top")
    }
  })

  output$risk_datatable <- DT::renderDataTable({
    filtered_risk() %>%
      arrange(desc(pct_games_missed_predicted)) %>%
      mutate(
        pct_games_missed_predicted = percent(pct_games_missed_predicted,
                                              accuracy = 0.1),
        prior_missed_pct           = percent(prior_missed_pct, accuracy = 0.1),
        avg_rolling_minutes_3      = round(avg_rolling_minutes_3, 1)
      ) %>%
      select(player_name, pct_games_missed_predicted,
             games_missed_of_82, risk_tier,
             avg_rolling_minutes_3, prior_missed_pct, career_season) %>%
      DT::datatable(
        options  = list(pageLength = 15, scrollX = TRUE,
                        order = list(list(1, "desc"))),
        rownames = FALSE,
        colnames = c("Player", "Predicted % missed", "Est. games missed",
                     "Risk tier", "Avg rolling mins",
                     "Prior missed %", "Career season")
      ) %>%
      DT::formatStyle(
        "risk_tier",
        backgroundColor = DT::styleEqual(
          c("High Risk", "Moderate Risk", "Low Risk"),
          c("#FDECEA",   "#FFF3E0",       "#E8F1FB")
        )
      )
  })

  ## Tab 3 — Bench the Instinct ---------------------------------------------

  whatif_result <- eventReactive(input$run_whatif, {
    whatif_predict(
      model                 = AI_MODEL_V2,
      avg_rolling_minutes_3 = input$wi_rolling_mins,
      load_spike            = input$wi_load_spike,
      prior_missed_pct      = input$wi_prior_missed,
      career_season         = input$wi_career,
      ever_injured          = as.integer(input$wi_ever_injured),
      total_rebounds        = input$wi_rebounds,
      total_points          = input$wi_points
    )
  })

  output$whatif_result_card <- renderUI({

    req(whatif_result())
    res <- whatif_result()

    tier_color <- switch(res$risk_tier,
      "High Risk"     = "#CE1141",
      "Moderate Risk" = "#FF8C00",
      "Low Risk"      = "#17408B"
    )

    tier_bg <- switch(res$risk_tier,
      "High Risk"     = "#FEF0F3",
      "Moderate Risk" = "#FFF8EE",
      "Low Risk"      = "#EEF3FB"
    )

    div(
      class = "result-card",
      style = glue("border-left: 6px solid {tier_color}; background: {tier_bg};"),

      fluidRow(
        column(4,
          p(class = "metric-label", "Risk tier"),
          h3(res$risk_tier,
             style = glue("color: {tier_color}; margin: 0;"))
        ),
        column(4,
          p(class = "metric-label", "Predicted % games missed"),
          p(class = "metric-big",
            style = glue("color: {tier_color};"),
            scales::percent(res$pct_games_missed_predicted, accuracy = 0.1))
        ),
        column(4,
          p(class = "metric-label", "Est. games missed (of 82)"),
          p(class = "metric-big",
            style = glue("color: {tier_color};"),
            glue("{res$games_missed_of_82} games"))
        )
      ),
      hr(),
      p(style = "color: #888; font-size: 0.85rem; margin: 0;",
        em(paste(
          "Adjust sliders and re-run to compare scenarios.",
          "Primary driver: avg rolling 3-game minutes (lagged).",
          "Increasing rolling minutes increases predicted absence."
        )))
    )
  })

  output$whatif_comparison_plot <- renderPlot({

    req(whatif_result())
    res <- whatif_result()

    risk_table_FILTERED %>%
      ggplot(aes(x = pct_games_missed_predicted)) +
      geom_histogram(bins = 30, fill = "#17408B",
                     alpha = 0.55, color = "white") +
      geom_vline(xintercept = res$pct_games_missed_predicted,
                 color = "#CE1141", linewidth = 2.5) +
      annotate(
        "label",
        x     = min(res$pct_games_missed_predicted + 0.04, 0.80),
        y     = Inf, vjust = 1.5,
        label = glue("Your player: {scales::percent(res$pct_games_missed_predicted, accuracy=0.1)}"),
        color = "#CE1141", fill = "white",
        size  = 4, fontface = "bold", hjust = 0,
        label.size = 0.5
      ) +
      scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                         limits = c(0, 1)) +
      labs(
        title    = "Where Does This Player Sit in the 2024 Cohort?",
        subtitle = "Red line = your simulated player | Blue = all meaningful contributors (≥20 rolling mins)",
        x        = "Predicted % games missed next season",
        y        = "Number of players",
        caption  = "Reference: 536 NBA players from 2024 regular season"
      ) +
      theme_nba()
  })

  ## Tab 4 — Model Info -----------------------------------------------------

  output$model_versions_table <- renderTable({
    model_version_comparison %>%
      rename(
        `Version`           = version,
        `Features removed`  = features_removed,
        `RMSE`              = rmse,
        `MAE`               = mae,
        `R²`                = rsq,
        `Note`              = note
      ) %>%
      mutate(across(c(RMSE, MAE, `R²`), ~ round(., 3)))
  }, striped = TRUE, hover = TRUE, bordered = TRUE, width = "100%")

  output$feature_imp_plot <- renderPlot({ feature_imp_VIZ })
  output$dist_plot        <- renderPlot({ pct_missed_dist_VIZ })
  output$cor_plot         <- renderPlot({ cor_VIZ })
}

###==========================================================================###
###  RUN                                                                       ###
###==========================================================================###

shinyApp(ui, server)
