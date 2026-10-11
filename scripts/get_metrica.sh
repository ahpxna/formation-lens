#!/usr/bin/env bash
# Metrica Sports open sample data (2 matches in CSV: tracking 25 fps + events).
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for g in Sample_Game_1 Sample_Game_2; do
  mkdir -p "data/metrica/data/$g"
  for f in RawTrackingData_Home_Team RawTrackingData_Away_Team RawEventsData; do
    out="data/metrica/data/$g/${g}_${f}.csv"
    [ -s "$out" ] || curl -fL --retry 3 -o "$out" "$base/$g/${g}_${f}.csv"
  done
done
du -sh data/metrica
