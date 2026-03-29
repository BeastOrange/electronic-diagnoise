# Final Thesis Summary

## Overview
- best_primary_score: 1.0000
- best_secondary_score: 1.0000
- num_benchmarks: 1
- num_datasets: 3
- num_runs: 51

## Key Findings
- Best overall primary score: cognitive_radio_spectrum / band / bagged_logistic_regression = 1.0000
- cognitive_radio_spectrum best: bagged_logistic_regression on band (accuracy=1.0000)
- vsb_power_line_fault best: random_forest on vsb_fault_detection_smoke (accuracy=0.9744)
- vsb_powerline_fault best: cnn_lstm on vsb_fault_cnn_lstm_main (f1=0.5394)
- Top accuracy: random_forest on burst (1.0000).
- Top macro_f1: random_forest on burst (1.0000).
- Tasks represented: 4
- Total benchmark runs: 48

## Dataset Notes
- Datasets represented: cognitive_radio_spectrum, vsb_power_line_fault, vsb_powerline_fault
- Run records collected: 51
- Prepared exploration directories collected: 4
- Benchmark directories collected: 1
- Figure manifest entries: 605
- Cross-dataset benchmark: best_dataset=cognitive_radio_spectrum, best_accuracy=1.0000, mean_accuracy=0.8370

## Output Files
- benchmark_metrics_merged.csv
- final_metrics.csv
- thesis_figures_manifest.csv
