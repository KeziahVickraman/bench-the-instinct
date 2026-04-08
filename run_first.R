###--------------------------------------------------------------------------###
###   run_first.R                                                             ###
###   Run this ONCE before rendering the Quarto website.                     ###
###   It generates the .RData files that all .qmd pages load from.          ###
###                                                                           ###
###   Step 1: Open RStudio and set working directory to this folder          ###
###   Step 2: Run this entire script                                         ###
###   Step 3: Then run: quarto render                                        ###
###--------------------------------------------------------------------------###

# Set working directory to this folder
# (If running interactively, use Session > Set Working Directory > To Source File Location)

cat("Step 1/3: Loading shared data...\n")
source("00_Data_Setup.R")

cat("Step 2/3: Running causal analysis (Part 1)...\n")
source("Part1_Causal_DiD.R")
# This saves: Part1_Causal_Results.RData

cat("Step 3/3: Running predictive model (Part 2)...\n")
# NOTE: Part 2 takes longer — it tunes XGBoost models
# If you have already run Part 2 and have Part2_Predictive_Results.RData,
# comment out the line below and it will load from the saved file instead
source("Part2_Predictive_Hurdle.R")
# This saves: Part2_Predictive_Results.RData

cat("\n✓ All done. You can now render the website:\n")
cat("  quarto render\n")
cat("\nOr preview it live:\n")
cat("  quarto preview\n")
