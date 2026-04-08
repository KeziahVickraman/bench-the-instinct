# Bench the Instinct — NBA Achilles Load Management
### Keziah Vickraman · Capstone 2025

## How to Run (3 steps)

**Step 1** — Open RStudio, set working directory to this folder:
`Session → Set Working Directory → To Source File Location`

**Step 2** — Generate the data files:
```r
source("run_first.R")
```

**Step 3** — Render the website (in RStudio Terminal tab):
```bash
quarto render
# or for live preview:
quarto preview
```

## Run the Shiny App
```r
shiny::runApp("Capstone_Shiny_PREDICTION.R")
```

## Deploy to GitHub Pages
```bash
git init && git add . && git commit -m "Initial"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
# GitHub → Settings → Pages → Branch: main, Folder: /docs
```

## Before going live, update two placeholders:
1. `YOUR_USERNAME/YOUR_REPO_NAME` in `_quarto.yml`
2. `YOUR_USERNAME.shinyapps.io/nba-achilles/` in `app.qmd`
