# Kaggle run report

models: `qwen2.5-3b, smollm2-1.7b, phi3.5-mini`
limit: full eval set
runtime: 401.4 min

## shim.log tail
```
INFO:     127.0.0.1:49718 - "POST /v1/verdict_probs HTTP/1.1" 200 OK
INFO:     127.0.0.1:44252 - "POST /v1/verdict_probs HTTP/1.1" 200 OK
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [39]
```

## baseline

| model | task | tp | fp | tn | fn | acc | prec | rec | f1 | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| phi3.5-mini | phishing_email | 120 | 14 | 186 | 80 | 0.7650 | 0.8955 | 0.6000 | 0.7186 | 0 |
| phi3.5-mini | sms_smishing | 163 | 54 | 146 | 37 | 0.7725 | 0.7512 | 0.8150 | 0.7818 | 0 |
| qwen2.5-3b | phishing_email | 191 | 23 | 177 | 9 | 0.9200 | 0.8925 | 0.9550 | 0.9227 | 0 |
| qwen2.5-3b | sms_smishing | 188 | 51 | 149 | 12 | 0.8425 | 0.7866 | 0.9400 | 0.8565 | 0 |
| smollm2-1.7b | phishing_email | 197 | 189 | 11 | 3 | 0.5200 | 0.5104 | 0.9850 | 0.6724 | 0 |
| smollm2-1.7b | sms_smishing | 185 | 175 | 25 | 15 | 0.5250 | 0.5139 | 0.9250 | 0.6607 | 0 |

## attack

ASR = malicious->benign flips / samples actually attacked. Only rows the clean model re-confirmed as malicious are counted. CI = 95% Wilson interval on the flip count.
| model | mode | task | attacked | flips | ASR | 95% CI | mean perturb | mean Jaccard |
|---|---|---|---|---|---|---|---|---|
| phi3.5-mini | char | phishing_email | 120 | 6 | 5.0% | 2.3-10.5% | 13.5% | 0.321 |
| phi3.5-mini | char | sms_smishing | 163 | 3 | 1.8% | 0.6-5.3% | 13.3% | 0.328 |
| qwen2.5-3b | char | phishing_email | 191 | 1 | 0.5% | 0.1-2.9% | 13.5% | 0.303 |
| qwen2.5-3b | char | sms_smishing | 188 | 0 | 0.0% | 0.0-2.0% | 13.5% | 0.322 |
| smollm2-1.7b | char | phishing_email | 197 | 1 | 0.5% | 0.1-2.8% | 13.4% | 0.311 |
| smollm2-1.7b | char | sms_smishing | 185 | 1 | 0.5% | 0.1-3.0% | 13.5% | 0.317 |
| phi3.5-mini | saliency | phishing_email | 120 | 14 | 11.7% | 7.1-18.6% | 6.0% | 0.867 |
| phi3.5-mini | saliency | sms_smishing | 163 | 12 | 7.4% | 4.3-12.4% | 12.6% | 0.773 |
| qwen2.5-3b | saliency | phishing_email | 169 | 4 | 2.4% | 0.9-5.9% | 5.4% | 0.876 |
| qwen2.5-3b | saliency | sms_smishing | 167 | 1 | 0.6% | 0.1-3.3% | 12.3% | 0.778 |
| qwen2.5-3b | saliency | (excluded: original not malicious) | 43 | - | - | - | - | - |
| smollm2-1.7b | saliency | phishing_email | 197 | 4 | 2.0% | 0.8-5.1% | 6.7% | 0.851 |
| smollm2-1.7b | saliency | sms_smishing | 185 | 11 | 5.9% | 3.4-10.3% | 12.9% | 0.768 |
| phi3.5-mini | word | phishing_email | 118 | 6 | 5.1% | 2.4-10.7% | 6.3% | 0.945 |
| phi3.5-mini | word | sms_smishing | 161 | 6 | 3.7% | 1.7-7.9% | 9.0% | 0.918 |
| phi3.5-mini | word | (excluded: original not malicious) | 4 | - | - | - | - | - |
| qwen2.5-3b | word | phishing_email | 190 | 5 | 2.6% | 1.1-6.0% | 6.2% | 0.950 |
| qwen2.5-3b | word | sms_smishing | 186 | 6 | 3.2% | 1.5-6.9% | 9.1% | 0.916 |
| qwen2.5-3b | word | (excluded: original not malicious) | 3 | - | - | - | - | - |
| smollm2-1.7b | word | phishing_email | 196 | 2 | 1.0% | 0.3-3.6% | 6.4% | 0.948 |
| smollm2-1.7b | word | sms_smishing | 184 | 3 | 1.6% | 0.6-4.7% | 8.8% | 0.916 |
| smollm2-1.7b | word | (excluded: original not malicious) | 2 | - | - | - | - | - |

## cross-model transfer

`flipped_source` = only perturbations that fooled the source model (the classic matrix, small n); `applied` = every perturbation the source applied, with the TARGET's own caught originals as the denominator.
| subset | source | target | mode | task | candidates | caught | flips | ASR | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| applied | phi3.5-mini | phi3.5-mini | char | phishing_email | 120 | 120 | 6 | 5.0% | 2.3-10.5% |
| applied | phi3.5-mini | qwen2.5-3b | char | phishing_email | 120 | 119 | 0 | 0.0% | 0.0-3.1% |
| applied | phi3.5-mini | smollm2-1.7b | char | phishing_email | 120 | 120 | 0 | 0.0% | 0.0-3.1% |
| applied | qwen2.5-3b | phi3.5-mini | char | phishing_email | 191 | 119 | 4 | 3.4% | 1.3-8.3% |
| applied | qwen2.5-3b | qwen2.5-3b | char | phishing_email | 191 | 191 | 1 | 0.5% | 0.1-2.9% |
| applied | qwen2.5-3b | smollm2-1.7b | char | phishing_email | 191 | 189 | 0 | 0.0% | 0.0-2.0% |
| applied | smollm2-1.7b | phi3.5-mini | char | phishing_email | 197 | 120 | 2 | 1.7% | 0.5-5.9% |
| applied | smollm2-1.7b | qwen2.5-3b | char | phishing_email | 197 | 189 | 1 | 0.5% | 0.1-2.9% |
| applied | smollm2-1.7b | smollm2-1.7b | char | phishing_email | 197 | 197 | 1 | 0.5% | 0.1-2.8% |
| applied | phi3.5-mini | phi3.5-mini | char | sms_smishing | 163 | 163 | 3 | 1.8% | 0.6-5.3% |
| applied | phi3.5-mini | qwen2.5-3b | char | sms_smishing | 163 | 163 | 0 | 0.0% | 0.0-2.3% |
| applied | phi3.5-mini | smollm2-1.7b | char | sms_smishing | 163 | 160 | 0 | 0.0% | 0.0-2.3% |
| applied | qwen2.5-3b | phi3.5-mini | char | sms_smishing | 188 | 163 | 1 | 0.6% | 0.1-3.4% |
| applied | qwen2.5-3b | qwen2.5-3b | char | sms_smishing | 188 | 188 | 0 | 0.0% | 0.0-2.0% |
| applied | qwen2.5-3b | smollm2-1.7b | char | sms_smishing | 188 | 179 | 0 | 0.0% | 0.0-2.1% |
| applied | smollm2-1.7b | phi3.5-mini | char | sms_smishing | 185 | 160 | 1 | 0.6% | 0.1-3.5% |
| applied | smollm2-1.7b | qwen2.5-3b | char | sms_smishing | 185 | 179 | 0 | 0.0% | 0.0-2.1% |
| applied | smollm2-1.7b | smollm2-1.7b | char | sms_smishing | 185 | 185 | 1 | 0.5% | 0.1-3.0% |
| applied | phi3.5-mini | phi3.5-mini | saliency | phishing_email | 120 | 120 | 14 | 11.7% | 7.1-18.6% |
| applied | phi3.5-mini | qwen2.5-3b | saliency | phishing_email | 120 | 119 | 1 | 0.8% | 0.1-4.6% |
| applied | phi3.5-mini | smollm2-1.7b | saliency | phishing_email | 120 | 120 | 0 | 0.0% | 0.0-3.1% |
| applied | qwen2.5-3b | phi3.5-mini | saliency | phishing_email | 169 | 113 | 4 | 3.5% | 1.4-8.8% |
| applied | qwen2.5-3b | qwen2.5-3b | saliency | phishing_email | 169 | 169 | 4 | 2.4% | 0.9-5.9% |
| applied | qwen2.5-3b | smollm2-1.7b | saliency | phishing_email | 169 | 167 | 1 | 0.6% | 0.1-3.3% |
| applied | smollm2-1.7b | phi3.5-mini | saliency | phishing_email | 197 | 120 | 5 | 4.2% | 1.8-9.4% |
| applied | smollm2-1.7b | qwen2.5-3b | saliency | phishing_email | 197 | 189 | 5 | 2.6% | 1.1-6.0% |
| applied | smollm2-1.7b | smollm2-1.7b | saliency | phishing_email | 197 | 197 | 4 | 2.0% | 0.8-5.1% |
| applied | phi3.5-mini | phi3.5-mini | saliency | sms_smishing | 163 | 163 | 12 | 7.4% | 4.3-12.4% |
| applied | phi3.5-mini | qwen2.5-3b | saliency | sms_smishing | 163 | 163 | 1 | 0.6% | 0.1-3.4% |
| applied | phi3.5-mini | smollm2-1.7b | saliency | sms_smishing | 163 | 160 | 0 | 0.0% | 0.0-2.3% |
| applied | qwen2.5-3b | phi3.5-mini | saliency | sms_smishing | 167 | 154 | 2 | 1.3% | 0.4-4.6% |
| applied | qwen2.5-3b | qwen2.5-3b | saliency | sms_smishing | 167 | 167 | 1 | 0.6% | 0.1-3.3% |
| applied | qwen2.5-3b | smollm2-1.7b | saliency | sms_smishing | 167 | 163 | 1 | 0.6% | 0.1-3.4% |
| applied | smollm2-1.7b | phi3.5-mini | saliency | sms_smishing | 185 | 160 | 0 | 0.0% | 0.0-2.3% |
| applied | smollm2-1.7b | qwen2.5-3b | saliency | sms_smishing | 185 | 179 | 4 | 2.2% | 0.9-5.6% |
| applied | smollm2-1.7b | smollm2-1.7b | saliency | sms_smishing | 185 | 185 | 11 | 5.9% | 3.4-10.3% |
| applied | phi3.5-mini | phi3.5-mini | word | phishing_email | 118 | 118 | 6 | 5.1% | 2.4-10.7% |
| applied | phi3.5-mini | qwen2.5-3b | word | phishing_email | 118 | 117 | 0 | 0.0% | 0.0-3.2% |
| applied | phi3.5-mini | smollm2-1.7b | word | phishing_email | 118 | 118 | 0 | 0.0% | 0.0-3.1% |
| applied | qwen2.5-3b | phi3.5-mini | word | phishing_email | 190 | 119 | 4 | 3.4% | 1.3-8.3% |
| applied | qwen2.5-3b | qwen2.5-3b | word | phishing_email | 190 | 190 | 5 | 2.6% | 1.1-6.0% |
| applied | qwen2.5-3b | smollm2-1.7b | word | phishing_email | 190 | 188 | 0 | 0.0% | 0.0-2.0% |
| applied | smollm2-1.7b | phi3.5-mini | word | phishing_email | 196 | 120 | 3 | 2.5% | 0.9-7.1% |
| applied | smollm2-1.7b | qwen2.5-3b | word | phishing_email | 196 | 188 | 1 | 0.5% | 0.1-2.9% |
| applied | smollm2-1.7b | smollm2-1.7b | word | phishing_email | 196 | 196 | 2 | 1.0% | 0.3-3.6% |
| applied | phi3.5-mini | phi3.5-mini | word | sms_smishing | 161 | 161 | 6 | 3.7% | 1.7-7.9% |
| applied | phi3.5-mini | qwen2.5-3b | word | sms_smishing | 161 | 161 | 1 | 0.6% | 0.1-3.4% |
| applied | phi3.5-mini | smollm2-1.7b | word | sms_smishing | 161 | 158 | 1 | 0.6% | 0.1-3.5% |
| applied | qwen2.5-3b | phi3.5-mini | word | sms_smishing | 186 | 162 | 3 | 1.8% | 0.6-5.3% |
| applied | qwen2.5-3b | qwen2.5-3b | word | sms_smishing | 186 | 186 | 6 | 3.2% | 1.5-6.9% |
| applied | qwen2.5-3b | smollm2-1.7b | word | sms_smishing | 186 | 178 | 0 | 0.0% | 0.0-2.1% |
| applied | smollm2-1.7b | phi3.5-mini | word | sms_smishing | 184 | 160 | 1 | 0.6% | 0.1-3.5% |
| applied | smollm2-1.7b | qwen2.5-3b | word | sms_smishing | 184 | 178 | 4 | 2.2% | 0.9-5.6% |
| applied | smollm2-1.7b | smollm2-1.7b | word | sms_smishing | 184 | 184 | 3 | 1.6% | 0.6-4.7% |
| flipped_source | phi3.5-mini | phi3.5-mini | char | phishing_email | 6 | 6 | 6 | 100.0% | 61.0-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | char | phishing_email | 6 | 6 | 0 | 0.0% | 0.0-39.0% |
| flipped_source | phi3.5-mini | smollm2-1.7b | char | phishing_email | 6 | 6 | 0 | 0.0% | 0.0-39.0% |
| flipped_source | qwen2.5-3b | qwen2.5-3b | char | phishing_email | 1 | 1 | 1 | 100.0% | 20.6-100.0% |
| flipped_source | qwen2.5-3b | smollm2-1.7b | char | phishing_email | 1 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | char | phishing_email | 1 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | char | phishing_email | 1 | 1 | 1 | 100.0% | 20.6-100.0% |
| flipped_source | phi3.5-mini | phi3.5-mini | char | sms_smishing | 3 | 3 | 3 | 100.0% | 43.9-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | char | sms_smishing | 3 | 3 | 0 | 0.0% | 0.0-56.1% |
| flipped_source | phi3.5-mini | smollm2-1.7b | char | sms_smishing | 3 | 3 | 0 | 0.0% | 0.0-56.1% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | char | sms_smishing | 1 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | char | sms_smishing | 1 | 1 | 1 | 100.0% | 20.6-100.0% |
| flipped_source | phi3.5-mini | phi3.5-mini | saliency | phishing_email | 14 | 14 | 14 | 100.0% | 78.5-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | saliency | phishing_email | 14 | 14 | 0 | 0.0% | 0.0-21.5% |
| flipped_source | phi3.5-mini | smollm2-1.7b | saliency | phishing_email | 14 | 14 | 0 | 0.0% | 0.0-21.5% |
| flipped_source | qwen2.5-3b | phi3.5-mini | saliency | phishing_email | 4 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | qwen2.5-3b | qwen2.5-3b | saliency | phishing_email | 4 | 4 | 4 | 100.0% | 51.0-100.0% |
| flipped_source | qwen2.5-3b | smollm2-1.7b | saliency | phishing_email | 4 | 4 | 0 | 0.0% | 0.0-49.0% |
| flipped_source | smollm2-1.7b | phi3.5-mini | saliency | phishing_email | 4 | 2 | 0 | 0.0% | 0.0-65.8% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | saliency | phishing_email | 4 | 4 | 2 | 50.0% | 15.0-85.0% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | saliency | phishing_email | 4 | 4 | 4 | 100.0% | 51.0-100.0% |
| flipped_source | phi3.5-mini | phi3.5-mini | saliency | sms_smishing | 12 | 12 | 12 | 100.0% | 75.8-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | saliency | sms_smishing | 12 | 12 | 1 | 8.3% | 1.5-35.4% |
| flipped_source | phi3.5-mini | smollm2-1.7b | saliency | sms_smishing | 12 | 9 | 0 | 0.0% | 0.0-29.9% |
| flipped_source | qwen2.5-3b | phi3.5-mini | saliency | sms_smishing | 1 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | qwen2.5-3b | qwen2.5-3b | saliency | sms_smishing | 1 | 1 | 1 | 100.0% | 20.6-100.0% |
| flipped_source | qwen2.5-3b | smollm2-1.7b | saliency | sms_smishing | 1 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | smollm2-1.7b | phi3.5-mini | saliency | sms_smishing | 11 | 5 | 0 | 0.0% | 0.0-43.5% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | saliency | sms_smishing | 11 | 11 | 1 | 9.1% | 1.6-37.7% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | saliency | sms_smishing | 11 | 11 | 11 | 100.0% | 74.1-100.0% |
| flipped_source | phi3.5-mini | phi3.5-mini | word | phishing_email | 6 | 6 | 6 | 100.0% | 61.0-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | word | phishing_email | 6 | 6 | 0 | 0.0% | 0.0-39.0% |
| flipped_source | phi3.5-mini | smollm2-1.7b | word | phishing_email | 6 | 6 | 0 | 0.0% | 0.0-39.0% |
| flipped_source | qwen2.5-3b | phi3.5-mini | word | phishing_email | 5 | 2 | 0 | 0.0% | 0.0-65.8% |
| flipped_source | qwen2.5-3b | qwen2.5-3b | word | phishing_email | 5 | 5 | 5 | 100.0% | 56.5-100.0% |
| flipped_source | qwen2.5-3b | smollm2-1.7b | word | phishing_email | 5 | 5 | 0 | 0.0% | 0.0-43.5% |
| flipped_source | smollm2-1.7b | phi3.5-mini | word | phishing_email | 2 | 1 | 0 | 0.0% | 0.0-79.3% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | word | phishing_email | 2 | 2 | 0 | 0.0% | 0.0-65.8% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | word | phishing_email | 2 | 2 | 2 | 100.0% | 34.2-100.0% |
| flipped_source | phi3.5-mini | phi3.5-mini | word | sms_smishing | 6 | 6 | 6 | 100.0% | 61.0-100.0% |
| flipped_source | phi3.5-mini | qwen2.5-3b | word | sms_smishing | 6 | 6 | 1 | 16.7% | 3.0-56.4% |
| flipped_source | phi3.5-mini | smollm2-1.7b | word | sms_smishing | 6 | 4 | 0 | 0.0% | 0.0-49.0% |
| flipped_source | qwen2.5-3b | phi3.5-mini | word | sms_smishing | 6 | 1 | 1 | 100.0% | 20.6-100.0% |
| flipped_source | qwen2.5-3b | qwen2.5-3b | word | sms_smishing | 6 | 6 | 6 | 100.0% | 61.0-100.0% |
| flipped_source | qwen2.5-3b | smollm2-1.7b | word | sms_smishing | 6 | 4 | 0 | 0.0% | 0.0-49.0% |
| flipped_source | smollm2-1.7b | phi3.5-mini | word | sms_smishing | 3 | 3 | 0 | 0.0% | 0.0-56.1% |
| flipped_source | smollm2-1.7b | qwen2.5-3b | word | sms_smishing | 3 | 3 | 0 | 0.0% | 0.0-56.1% |
| flipped_source | smollm2-1.7b | smollm2-1.7b | word | sms_smishing | 3 | 3 | 3 | 100.0% | 43.9-100.0% |

self-replay: 18 cell(s) covering 3080 replayed rows, every flip count matches the attack run.

## saliency verdict-channel check

The saliency attack decides success on P(malicious) < 0.5; every other mode uses the generated verdict. Both counts are shown because a gap between them is a result about the measurement, not a bug.
| model | task | n | flips (prob) | flips (gen) | agree | prob-only | gen-only |
|---|---|---|---|---|---|---|---|
| phi3.5-mini | phishing_email | 120 | 14 | 14 | 120 | 0 | 0 |
| phi3.5-mini | sms_smishing | 163 | 12 | 12 | 163 | 0 | 0 |
| qwen2.5-3b | phishing_email | 169 | 28 | 4 | 145 | 24 | 0 |
| qwen2.5-3b | sms_smishing | 167 | 22 | 1 | 146 | 21 | 0 |
| smollm2-1.7b | phishing_email | 197 | 0 | 4 | 193 | 0 | 4 |
| smollm2-1.7b | sms_smishing | 185 | 0 | 11 | 174 | 0 | 11 |

## defence operating points

tau is fitted on clean benign mail to the stated FPR budget; clean TPR is what the filter costs.
| model | task | tau | FPR @0.5 | FPR @tau | TPR @0.5 | TPR @tau | budget met |
|---|---|---|---|---|---|---|---|
| phi3.5-mini | phishing_email | 0.9604 | 0.070 | 0.050 | 0.600 | 0.580 | True |
| phi3.5-mini | sms_smishing | 1.0000 | 0.270 | 0.150 | 0.815 | 0.740 | False |
| qwen2.5-3b | phishing_email | 0.8222 | 0.070 | 0.050 | 0.845 | 0.765 | True |
| qwen2.5-3b | sms_smishing | 0.8459 | 0.115 | 0.050 | 0.835 | 0.745 | True |
| smollm2-1.7b | phishing_email | 0.9860 | 1.000 | 0.050 | 1.000 | 0.345 | True |
| smollm2-1.7b | sms_smishing | 0.9901 | 1.000 | 0.050 | 1.000 | 0.230 | True |

## files
```
results/eval_subset.csv                       447867 bytes
results/attack_eligible.csv                   565642 bytes
results/clean_baseline_results.csv            136702 bytes
results/baseline_table.csv                    492 bytes
results/attack_ch1_input.csv                            34707 bytes
results/attack_eligible.csv                             565642 bytes
results/attack_eligible_phi3.5-mini_phishing_email.csv  89537 bytes
results/attack_eligible_phi3.5-mini_sms_smishing.csv    33224 bytes
results/attack_eligible_qwen2.5-3b_phishing_email.csv   183234 bytes
results/attack_eligible_qwen2.5-3b_sms_smishing.csv     37489 bytes
results/attack_eligible_smollm2-1.7b_phishing_email.csv 185259 bytes
results/attack_eligible_smollm2-1.7b_sms_smishing.csv   37184 bytes
results/attack_results_char_phi3.5-mini.csv             142140 bytes
results/attack_results_char_qwen2.5-3b.csv              254330 bytes
results/attack_results_char_smollm2-1.7b.csv            255460 bytes
results/attack_results_saliency_phi3.5-mini.csv         164361 bytes
results/attack_results_saliency_qwen2.5-3b.csv          258006 bytes
results/attack_results_saliency_smollm2-1.7b.csv        282673 bytes
results/attack_results_word_phi3.5-mini.csv             130305 bytes
results/attack_results_word_qwen2.5-3b.csv              226384 bytes
results/attack_results_word_smollm2-1.7b.csv            227914 bytes
results/attack_table.csv                                2193 bytes
results/asr_summary_char_phi3.5-mini.txt                408 bytes
results/asr_summary_char_qwen2.5-3b.txt                 407 bytes
results/asr_summary_char_smollm2-1.7b.txt               409 bytes
results/asr_summary_saliency_phi3.5-mini.txt            455 bytes
results/asr_summary_saliency_qwen2.5-3b.txt             457 bytes
results/asr_summary_saliency_smollm2-1.7b.txt           454 bytes
results/asr_summary_word_phi3.5-mini.txt                408 bytes
results/asr_summary_word_qwen2.5-3b.txt                 407 bytes
results/asr_summary_word_smollm2-1.7b.txt               408 bytes
```
